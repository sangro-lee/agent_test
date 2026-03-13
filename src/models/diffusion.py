from __future__ import annotations

import torch
from torch import nn


class DenoisingMLP(nn.Module):
    """
    Unconditional denoising network.
    Input: (z_t, t_normalized) → predicted noise epsilon
    """
    def __init__(self, latent_dim: int = 200, time_dim: int = 64, hidden_dim: int = 512):
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


class ConditionalDenoisingMLP(nn.Module):
    """
    Classifier-Free Guidance (CFG) denoising network.
    Input: (z_t, t_normalized, c) where c = normalized pIC50 scalar (or null=0 for uncond)

    CFG inference:
        eps_uncond = forward(z_t, t, c=null)
        eps_cond   = forward(z_t, t, c=pIC50)
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    """
    NULL_COND = 0.0  # sentinel for unconditional (dropped condition)

    def __init__(
        self,
        latent_dim: int = 200,
        time_dim: int = 64,
        cond_dim: int = 64,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.time_dim = int(time_dim)
        self.cond_dim = int(cond_dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        # Condition embedding (scalar pIC50 → cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        self.main = nn.Sequential(
            nn.Linear(self.latent_dim + self.time_dim + self.cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

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

        t_embed = self.time_mlp(t)
        c_embed = self.cond_mlp(c)
        h = torch.cat([z_t, t_embed, c_embed], dim=1)
        return self.main(h)


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
