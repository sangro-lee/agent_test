from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal


class SinusoidalTimeEmbedding(nn.Module):
    """
    Drop-in replacement for time_mlp. Matches G2D-Diff get_timestep_embedding + time_layer.
    Accepts same input as current time_mlp: t (B, 1) normalized [0, 1].
    Usage: self.time_mlp = SinusoidalTimeEmbedding(time_dim)  — forward unchanged.
    """
    def __init__(self, time_dim: int, T: int = 1000):
        super().__init__()
        self.half = time_dim // 2
        self.T = T
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_int = t * self.T
        freq = torch.exp(
            -math.log(10000) *
            torch.arange(self.half, dtype=torch.float32, device=t.device) / (self.half - 1)
        )
        emb = t_int * freq
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class DenoisingMLP(nn.Module):
    """
    Unconditional denoising network.
    Input: (z_t, t_normalized) → predicted noise epsilon
    """
    def __init__(self, latent_dim: int = 128, time_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.time_dim = int(time_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.main = nn.Sequential(
            nn.Linear(self.latent_dim + self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

        self.register_buffer("z_mean", torch.zeros(self.latent_dim), persistent=True)
        self.register_buffer("z_std", torch.ones(self.latent_dim), persistent=True)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.z_mean.copy_(mean.detach())
        self.z_std.copy_(std.detach())

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_embed = self.time_mlp(t)
        h = torch.cat([z_t, t_embed], dim=1)
        return self.main(h)


class _CondInj(nn.Module):
    """
    Simple AdaIN condition injection: x += w * h_norm + b.
    Direct projection from cond_embed_dim to hidden_dim*2 — no bottleneck.
    Shared by MLP and VQC.
    """
    def __init__(self, hidden_dim: int, cond_embed_dim: int):
        super().__init__()
        self.cond2wb = nn.Linear(cond_embed_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, h_norm: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        w, b = self.cond2wb(cond).chunk(2, dim=-1)
        return x + w * h_norm + b


class ConditionalDenoisingMLP(nn.Module):
    """
    Classifier-Free Guidance (CFG) denoising network.
    Condition is injected at every layer via AdaIN-style scale/shift
    (following G2D-Diff: x_{l+1} = f2(GELU(w_c * norm(f1(x_l)) + b_c))).

    CFG inference:
        eps_uncond = forward(z_t, t, c=null)
        eps_cond   = forward(z_t, t, c=pIC50)
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    """
    NULL_COND = 0.0  # sentinel for unconditional (dropped condition)

    def __init__(
        self,
        latent_dim: int = 128,
        time_dim: int = 128,
        cond_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 6,
        use_orthogonal: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.time_dim = int(time_dim)
        self.cond_dim = int(cond_dim)
        self.num_layers = int(num_layers)
        self.use_orthogonal = bool(use_orthogonal)

        # To switch to sinusoidal embedding (G2D-Diff style), replace with:
        # self.time_mlp = SinusoidalTimeEmbedding(self.time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        cond_embed_dim = self.time_dim + self.cond_dim

        # Per-block: latent → hidden → latent, then CondInj at latent_dim
        self.input_projs = nn.ModuleList([
            nn.Linear(self.latent_dim, hidden_dim) for _ in range(self.num_layers)
        ])
        self.f1_layers = nn.ModuleList([
            nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.latent_dim) for _ in range(self.num_layers)
        ])
        self.output_projs = nn.ModuleList([
            nn.Linear(hidden_dim, self.latent_dim) for _ in range(self.num_layers)
        ])
        self.cond_layers = nn.ModuleList([
            _CondInj(self.latent_dim, cond_embed_dim) for _ in range(self.num_layers)
        ])

        self.final_layer = nn.Linear(self.latent_dim, self.latent_dim)

        if use_orthogonal:
            for f1 in self.f1_layers:
                orthogonal(f1)

        self.register_buffer("z_mean", torch.zeros(self.latent_dim), persistent=True)
        self.register_buffer("z_std", torch.ones(self.latent_dim), persistent=True)
        self.register_buffer("c_mean", torch.zeros(1), persistent=True)
        self.register_buffer("c_std", torch.ones(1), persistent=True)

    def set_normalization(self, z_mean, z_std, c_mean=None, c_std=None):
        self.z_mean.copy_(z_mean.detach())
        self.z_std.copy_(z_std.detach())
        if c_mean is not None:
            self.c_mean.copy_(c_mean.detach().view(1))
        if c_std is not None:
            self.c_std.copy_(c_std.detach().view(1))

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: (B, latent_dim)
            t:   (B,) or (B, 1), normalized to [0, 1]
            c:   (B, 1), normalized pIC50. Use 0.0 (NULL_COND) for unconditional.
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if c.dim() == 1:
            c = c.unsqueeze(1)

        t_embed = self.time_mlp(t)                        # (B, time_dim)
        c_embed = self.cond_mlp(c)                        # (B, cond_dim)
        cond = torch.cat([t_embed, c_embed], dim=-1)      # (B, time_dim+cond_dim)

        x = z_t
        for f1, norm, cond_layer in zip(self.f1_layers, self.norms, self.cond_layers):
            h = norm(f1(x))              # latent_dim throughout
            x = cond_layer(x, h, cond)
#        return self.final_layer(x)
        return x

class NoiseScheduler:
    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ):
        self.T = int(T)
        self.device = torch.device(device)

        self.betas = torch.linspace(beta_start, beta_end, self.T, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        alpha_bars_prev = torch.cat([
            torch.ones(1, device=self.device),
            self.alpha_bars[:-1],
        ])
        self.posterior_var = self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.posterior_var = torch.clamp(self.posterior_var, min=1e-20)

    def to(self, device: str | torch.device) -> "NoiseScheduler":
        return NoiseScheduler(
            T=self.T,
            beta_start=float(self.betas[0].item()),
            beta_end=float(self.betas[-1].item()),
            device=device,
        )

    def _gather(self, arr: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        out = arr[t].view(-1, 1)
        return out.to(device=like.device, dtype=like.dtype)

    def add_noise(self, z0: torch.Tensor, t: torch.Tensor):
        eps = torch.randn_like(z0)
        sqrt_ab = self._gather(self.sqrt_alpha_bars, t, z0)
        sqrt_1mab = self._gather(self.sqrt_one_minus_alpha_bars, t, z0)
        z_t = sqrt_ab * z0 + sqrt_1mab * eps
        return z_t, eps

    def ddpm_step(self, z_t: torch.Tensor, t_int: int, eps_pred: torch.Tensor) -> torch.Tensor:
        t_idx = int(t_int) - 1
        if t_idx < 0:
            return z_t

        alpha_t = self.alphas[t_idx].to(z_t.device, z_t.dtype)
        alpha_bar_t = self.alpha_bars[t_idx].to(z_t.device, z_t.dtype)
        beta_t = self.betas[t_idx].to(z_t.device, z_t.dtype)

        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (z_t - coef * eps_pred)

        if t_idx == 0:
            return mean

        var = self.posterior_var[t_idx].to(z_t.device, z_t.dtype)
        noise = torch.randn_like(z_t)
        return mean + torch.sqrt(var) * noise

    def ddim_step(
        self,
        z_t: torch.Tensor,
        t_int: int,
        eps_pred: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        t_idx = int(t_int) - 1
        if t_idx < 0:
            return z_t

        alpha_bar_t = self.alpha_bars[t_idx].to(z_t.device, z_t.dtype)
        if t_idx == 0:
            alpha_bar_prev = torch.ones((), device=z_t.device, dtype=z_t.dtype)
        else:
            alpha_bar_prev = self.alpha_bars[t_idx - 1].to(z_t.device, z_t.dtype)

        x0_pred = (z_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        sigma = (
            eta
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
        )
        dir_term = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma * sigma, min=0.0)) * eps_pred

        if float(eta) > 0.0 and t_idx > 0:
            noise = torch.randn_like(z_t)
        else:
            noise = torch.zeros_like(z_t)

        z_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_term + sigma * noise
        if t_idx == 0:
            return x0_pred
        return z_prev
