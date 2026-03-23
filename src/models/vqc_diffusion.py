"""
Hybrid U-Net denoiser for latent diffusion with optional VQC bottleneck.

Architecture:
    z_t(B, latent_dim), t(B,1), c(B,1)
        cond = cat([time_mlp(t), cond_mlp(c)])
        │
        Enc Block 1 (latent_dim → unet_dims[0]) ────────── skip h1
        Enc Block 2 (unet_dims[0] → unet_dims[1]) ────────── skip h2
        Enc Block 3 (unet_dims[1] → unet_dims[2]) ────────── skip h3
        │
        [VQC Bottleneck] use_vqc=True:
          normalize(bottleneck_dim) → AmplitudeEmbedding(n_qubits = log2(bottleneck_dim))
          VQC circuit → [expval(Z(i))] → proj_out(n_qubits → bottleneck_dim)
        OR [Classical Bottleneck] use_vqc=False: _CondInjectionLayer
        + _CondInjectionLayer(cond)
        │
        Dec Block 1: cat(hb ‖ h3) → unet_dims[1] + _CondInjectionLayer
        Dec Block 2: cat(d1 ‖ h2) → unet_dims[0] + _CondInjectionLayer
        Dec Block 3: cat(d2 ‖ h1) → latent_dim  + _CondInjectionLayer
        out_head: Linear(latent_dim → latent_dim) [no activation]
        ε̂ (B, latent_dim)

Skip connections bypass the VQC bottleneck so information is preserved
even if the quantum circuit compresses or loses some signal.
VQC output (n_qubits-dim) is always reprojected to bottleneck_dim before
the decoder, since the decoder expects full-dimensional feature maps.
"""
from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn

from src.models.diffusion import _CondInjectionLayer


class HybridUNetDenoiser(nn.Module):
    """
    U-Net denoiser for latent CFG diffusion, compatible with
    ConditionalDenoisingMLP interface (NULL_COND, set_normalization, forward).

    Args:
        latent_dim:   Dimension of the latent vectors (input/output).
        unet_dims:    Hidden dims of the U-Net encoder, e.g. [128, 64, 32].
                      The last element is the bottleneck dim (must be a power
                      of 2 when use_vqc=True, e.g. 32 → 5 qubits).
        n_layers:     Number of parameterized VQC layers (ignored if use_vqc=False).
        time_dim:     Timestep embedding size.
        cond_dim:     Condition (pIC50) embedding size.
        use_vqc:      If True, VQC replaces the classical bottleneck.
        device_type:  PennyLane device (e.g. "default.qubit", "lightning.gpu").
    """

    NULL_COND = 0.0

    def __init__(
        self,
        latent_dim: int = 256,
        unet_dims: list | None = None,
        n_layers: int = 2,
        time_dim: int = 128,
        cond_dim: int = 128,
        use_vqc: bool = True,
        device_type: str = "default.qubit",
    ):
        super().__init__()
        if unet_dims is None:
            unet_dims = [128, 64, 32]

        self.latent_dim = int(latent_dim)
        self.unet_dims = [int(d) for d in unet_dims]
        self.n_layers = int(n_layers)
        self.use_vqc = bool(use_vqc)

        cond_embed_dim = int(time_dim) + int(cond_dim)

        # ── Condition embedding (same pattern as ConditionalDenoisingMLP) ──
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        # ── Encoder blocks ──────────────────────────────────────────────────
        dims = [self.latent_dim] + self.unet_dims
        self.enc_projs = nn.ModuleList()
        self.enc_conds = nn.ModuleList()
        for i in range(len(self.unet_dims)):
            self.enc_projs.append(nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.GELU()))
            self.enc_conds.append(_CondInjectionLayer(dims[i + 1], cond_embed_dim))

        # ── Bottleneck ───────────────────────────────────────────────────────
        bottleneck_dim = self.unet_dims[-1]  # e.g. 64

        if use_vqc:
            n_qubits = int(round(math.log2(bottleneck_dim)))
            assert 2 ** n_qubits == bottleneck_dim, (
                f"unet_dims[-1]={bottleneck_dim} must be a power of 2 "
                f"for AmplitudeEmbedding (got log2={n_qubits:.3f})"
            )
            self.n_qubits = n_qubits
            self.bottleneck_dim = bottleneck_dim

            dev = qml.device(device_type, wires=n_qubits)
            # Ring topology: n_qubits rotations + n_qubits entanglement pairs per layer
            params_per_layer = (n_qubits * 3) + (n_qubits * 3)
            weight_shapes = {"theta": (params_per_layer * n_layers,)}

            _pi = math.pi  # capture for use inside qnode

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def vqc_circuit(inputs, theta):
                # Amplitude encoding: 2^n_qubits = bottleneck_dim
                qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=False)

                # Initial entanglement (ring: last qubit → first qubit)
                for j in range(n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % n_qubits])

                param_idx = 0
                for _layer in range(n_layers):
                    # Rotation sub-layer
                    for i in range(n_qubits):
                        qml.RZ(theta[param_idx],     wires=i)
                        qml.RY(theta[param_idx + 1], wires=i)
                        qml.RZ(theta[param_idx + 2], wires=i)
                        param_idx += 3
                    # Entanglement sub-layer (ring: n_qubits pairs, last wraps to first)
                    for p in range(n_qubits):
                        nxt = (p + 1) % n_qubits
                        qml.RZ(-_pi / 2,             wires=nxt)
                        qml.CNOT(wires=[nxt, p])
                        qml.RZ(theta[param_idx],     wires=p)
                        qml.RY(theta[param_idx + 1], wires=nxt)
                        qml.CNOT(wires=[p, nxt])
                        qml.RY(theta[param_idx + 2], wires=nxt)
                        param_idx += 3

                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self.vqc = qml.qnn.TorchLayer(vqc_circuit, weight_shapes)
            # Reproject quantum measurements (n_qubits-dim) back to bottleneck_dim
            self.bottleneck_proj_out = nn.Linear(n_qubits, bottleneck_dim)

        # Condition injection applied after bottleneck (classical or VQC)
        self.bottleneck_cond = _CondInjectionLayer(bottleneck_dim, cond_embed_dim)

        # ── Decoder blocks ───────────────────────────────────────────────────
        # rev_dims[i] = unet_dims[-(i+1)], e.g. [64, 128, 256]
        rev_dims = list(reversed(self.unet_dims))
        self.dec_projs = nn.ModuleList()
        self.dec_conds = nn.ModuleList()
        for i in range(len(rev_dims)):
            in_dim  = rev_dims[i] * 2                                           # cat with skip
            out_dim = rev_dims[i + 1] if i + 1 < len(rev_dims) else self.latent_dim
            self.dec_projs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU()))
            self.dec_conds.append(_CondInjectionLayer(out_dim, cond_embed_dim))

        # ── Output head (no activation) ─────────────────────────────────────
        self.out_head = nn.Linear(self.latent_dim, self.latent_dim)

        # ── Normalization buffers (same interface as ConditionalDenoisingMLP) ─
        self.register_buffer("z_mean", torch.zeros(self.latent_dim), persistent=True)
        self.register_buffer("z_std",  torch.ones(self.latent_dim),  persistent=True)
        self.register_buffer("c_mean", torch.zeros(1),               persistent=True)
        self.register_buffer("c_std",  torch.ones(1),                persistent=True)

    def set_normalization(
        self,
        z_mean: torch.Tensor,
        z_std:  torch.Tensor,
        c_mean: torch.Tensor | None = None,
        c_std:  torch.Tensor | None = None,
    ):
        self.z_mean.copy_(z_mean.detach())
        self.z_std.copy_(z_std.detach())
        if c_mean is not None:
            self.c_mean.copy_(c_mean.detach().view(1))
        if c_std is not None:
            self.c_std.copy_(c_std.detach().view(1))

    def _vqc_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """AmplitudeEmbedding → VQC → proj_out. x: (B, bottleneck_dim)."""
        x_d = x.double()
        x_d = torch.nan_to_num(x_d, nan=0.0, posinf=0.0, neginf=0.0)
        norms = x_d.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        x_d = x_d / norms                    # unit-norm for AmplitudeEmbedding
        q_out = self.vqc(x_d)               # (B, n_qubits)
        return self.bottleneck_proj_out(q_out.float())  # (B, bottleneck_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t:   torch.Tensor,
        c:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t: (B, latent_dim)  noisy latent at timestep t
            t:   (B,) or (B, 1)  normalized timestep in [0, 1]
            c:   (B, 1)          normalized pIC50; use NULL_COND for unconditional
        Returns:
            eps_pred: (B, latent_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if c.dim() == 1:
            c = c.unsqueeze(1)

        t_emb = self.time_mlp(t)                        # (B, time_dim)
        c_emb = self.cond_mlp(c)                        # (B, cond_dim)
        cond  = torch.cat([t_emb, c_emb], dim=-1)      # (B, cond_embed_dim)

        # ── Encoder ───────────────────────────────────────────────────────
        skips = []
        x = z_t
        for proj, cond_layer in zip(self.enc_projs, self.enc_conds):
            x = proj(x)
            x = cond_layer(x, cond)
            skips.append(x)                             # save for skip connection

        # ── Bottleneck ────────────────────────────────────────────────────
        if self.use_vqc:
            x = self._vqc_bottleneck(x)                # quantum transform
        x = self.bottleneck_cond(x, cond)              # condition injection

        # ── Decoder (reverse skips) ───────────────────────────────────────
        for proj, cond_layer, skip in zip(
            self.dec_projs, self.dec_conds, reversed(skips)
        ):
            x = torch.cat([x, skip], dim=-1)           # skip connection
            x = proj(x)
            x = cond_layer(x, cond)

        return self.out_head(x)                         # (B, latent_dim)


class AngleVQCDenoiser(nn.Module):
    """
    Pure VQC denoiser using angle embedding for small-latent diffusion.

    n_qubits = latent_dim (one qubit per feature dimension).
    No amplitude embedding — uses RY angle encoding, so latent_dim can be
    any integer (no power-of-2 constraint).

    Forward path:
        z_t(B, D), t(B,1), c(B,1)
            time_mlp(t) → t_emb(B, time_dim)
            cond_mlp(c) → c_emb(B, cond_dim)
            cat([z_t, t_emb, c_emb]) → pre_proj → tanh → (B, D)  ← angles in [-1,1]
            VQC: AngleEmbedding(π * x, RY) + n_layers × (RZY + ring CNOT)
            [expval(Z_i)] → (B, D)
            post_affine: δ1 * q_out + δ2 → ε̂(B, D)  (per-dim scale+bias)
    """

    NULL_COND = 0.0

    def __init__(
        self,
        latent_dim:  int = 8,
        n_layers:    int = 2,
        time_dim:    int = 32,
        cond_dim:    int = 32,
        device_type: str = "default.qubit",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_layers   = int(n_layers)
        self.time_dim   = int(time_dim)
        self.cond_dim   = int(cond_dim)

        n_qubits = self.latent_dim

        # ── Classical conditioning ────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        # ── Pre-projection: (z_t ‖ t_emb ‖ c_emb) → latent_dim ─────────
        self.pre_proj = nn.Sequential(
            nn.Linear(n_qubits + time_dim + cond_dim, n_qubits),
            nn.Tanh(),   # bound to (-1, 1) → angles π*x ∈ (-π, π)
        )

        # ── VQC (angle embedding) ────────────────────────────────────────
        _pi = math.pi
        dev = qml.device(device_type, wires=n_qubits)
        params_per_layer = n_qubits * 3 + n_qubits * 3   # rot + entangle
        weight_shapes = {"theta": (params_per_layer * n_layers,)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def vqc_circuit(inputs, theta):
            # Angle encoding: RY(π * x_i) per qubit
            qml.AngleEmbedding(inputs * _pi, wires=range(n_qubits), rotation="Y")

            param_idx = 0
            for _layer in range(n_layers):
                # Rotation sub-layer
                for i in range(n_qubits):
                    qml.RZ(theta[param_idx],     wires=i)
                    qml.RY(theta[param_idx + 1], wires=i)
                    qml.RZ(theta[param_idx + 2], wires=i)
                    param_idx += 3
                # Ring entanglement
                for p in range(n_qubits):
                    nxt = (p + 1) % n_qubits
                    qml.RZ(-_pi / 2,             wires=nxt)
                    qml.CNOT(wires=[nxt, p])
                    qml.RZ(theta[param_idx],     wires=p)
                    qml.RY(theta[param_idx + 1], wires=nxt)
                    qml.CNOT(wires=[p, nxt])
                    qml.RY(theta[param_idx + 2], wires=nxt)
                    param_idx += 3

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.vqc = qml.qnn.TorchLayer(vqc_circuit, weight_shapes)

        # ── Post-affine: δ1 * measurement + δ2  (per-dimension) ────────────
        # Simpler than Linear(D→D): respects qubit independence, 2D params vs D²
        self.post_scale = nn.Parameter(torch.ones(self.latent_dim))
        self.post_bias  = nn.Parameter(torch.zeros(self.latent_dim))

        # ── Normalization buffers ─────────────────────────────────────────
        self.register_buffer("z_mean", torch.zeros(self.latent_dim), persistent=True)
        self.register_buffer("z_std",  torch.ones(self.latent_dim),  persistent=True)
        self.register_buffer("c_mean", torch.zeros(1),               persistent=True)
        self.register_buffer("c_std",  torch.ones(1),                persistent=True)

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
            t:   (B,) or (B, 1)  normalized timestep in [0, 1]
            c:   (B, 1)          normalized pIC50; use NULL_COND for unconditional
        Returns:
            eps_pred: (B, latent_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if c.dim() == 1:
            c = c.unsqueeze(1)

        t_emb = self.time_mlp(t)                          # (B, time_dim)
        c_emb = self.cond_mlp(c)                          # (B, cond_dim)

        x = self.pre_proj(torch.cat([z_t, t_emb, c_emb], dim=-1))  # (B, n_qubits), tanh

        # VQC expects float64 for numerical stability
        q_out = self.vqc(x.double()).float()              # (B, n_qubits)

        return self.post_scale * q_out + self.post_bias   # (B, latent_dim)
