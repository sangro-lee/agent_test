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

    def to(self, *args, **kwargs):
        """Keep VQC circuit on CPU regardless of device transfer."""
        super().to(*args, **kwargs)
        if self.use_vqc:
            self.vqc.cpu()
        return self

    def _vqc_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """AmplitudeEmbedding → VQC → proj_out. x: (B, bottleneck_dim)."""
        x_d = x.double()
        x_d = torch.nan_to_num(x_d, nan=0.0, posinf=0.0, neginf=0.0)
        norms = x_d.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        x_d = x_d / norms                    # unit-norm for AmplitudeEmbedding
        q_out = self.vqc(x_d.cpu()).to(x.device)  # VQC on CPU (statevector), move back
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


def _make_reupload_vqc_layer(
    n_qubits: int,
    n_layers: int,
    device_type: str,
) -> "qml.qnn.TorchLayer":
    """Factory for data re-uploading VQC blocks (Skolik architecture).

    inputs shape: (n_layers * n_qubits,) — pre-scaled by lambda_scales outside the qnode.
    At each layer l, inputs[l*n_qubits:(l+1)*n_qubits] are re-encoded via AngleEmbedding.
    """
    _pi = math.pi
    dev = qml.device(device_type, wires=n_qubits)
    params_per_layer = n_qubits * 3 + n_qubits * 3  # rot + entangle
    weight_shapes = {"theta": (params_per_layer * n_layers,)}

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def vqc_circuit(inputs, theta):
        # inputs: (n_layers * n_qubits,) pre-scaled by λ
        param_idx = 0
        for l in range(n_layers):
            # Data re-uploading: encode at the start of EVERY layer
            x_l = inputs[l * n_qubits : (l + 1) * n_qubits]
            qml.AngleEmbedding(x_l * _pi, wires=range(n_qubits), rotation="Y")
            # Variational rotations
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

    return qml.qnn.TorchLayer(vqc_circuit, weight_shapes)


def _make_angle_vqc_layer(
    n_qubits: int,
    n_layers: int,
    device_type: str,
) -> "qml.qnn.TorchLayer":
    """Factory to create an independent TorchLayer for each VQC block."""
    _pi = math.pi
    dev = qml.device(device_type, wires=n_qubits)
    params_per_layer = n_qubits * 3 + n_qubits * 3  # rot + entangle
    weight_shapes = {"theta": (params_per_layer * n_layers,)}

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def vqc_circuit(inputs, theta):
        qml.AngleEmbedding(inputs * _pi, wires=range(n_qubits), rotation="Y")
        param_idx = 0
        for _ in range(n_layers):
            for i in range(n_qubits):
                qml.RZ(theta[param_idx],     wires=i)
                qml.RY(theta[param_idx + 1], wires=i)
                qml.RZ(theta[param_idx + 2], wires=i)
                param_idx += 3
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

    return qml.qnn.TorchLayer(vqc_circuit, weight_shapes)


class AngleVQCDenoiser(nn.Module):
    """
    VQC denoiser with num_blocks × [AngleVQC + CondInj], mirroring ConditionalDenoisingMLP.

    Each block:
        q_out = VQC_i(AngleEmbedding(π * x))        # angle-encode x, run circuit
        w, b  = cond2wb_i(cat[t_emb, c_emb])        # dynamic scale/shift from t, c
        x     = x + w * LayerNorm(q_out) + b         # residual + AdaIN

    z_t enters the first block directly (no pre_proj distortion).
    t/c conditioning is injected at every block, matching the MLP pattern.
    """

    NULL_COND = 0.0

    def __init__(
        self,
        latent_dim:   int = 8,
        n_layers:     int = 2,       # circuit depth per VQC block
        num_blocks:   int = 6,       # number of [VQC + CondInj] blocks (matches MLP num_layers)
        time_dim:     int = 32,
        cond_dim:     int = 32,
        device_type:  str = "default.qubit",
        use_delta:    bool = False,  # per-block learnable affine on VQC output (δ1·norm(q)+δ2)
        use_reupload: bool = False,  # data re-uploading: encode x at start of every layer
    ):
        super().__init__()
        self.latent_dim   = int(latent_dim)
        self.n_layers     = int(n_layers)
        self.num_blocks   = int(num_blocks)
        self.time_dim     = int(time_dim)
        self.cond_dim     = int(cond_dim)
        self.use_delta    = bool(use_delta)
        self.use_reupload = bool(use_reupload)

        n_qubits       = self.latent_dim
        cond_embed_dim = time_dim + cond_dim

        # ── Classical conditioning embeddings ─────────────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        # ── num_blocks independent VQC circuits ──────────────────────────
        if use_reupload:
            self.vqc_blocks = nn.ModuleList([
                _make_reupload_vqc_layer(n_qubits, int(n_layers), device_type)
                for _ in range(num_blocks)
            ])
        else:
            self.vqc_blocks = nn.ModuleList([
                _make_angle_vqc_layer(n_qubits, int(n_layers), device_type)
                for _ in range(num_blocks)
            ])

        # ── CondInj per block: cond → (w, b) each of size D ──────────────
        # Direct projection (no bottleneck) to preserve conditioning signal.
        self.cond2wb = nn.ModuleList([
            nn.Linear(cond_embed_dim, n_qubits * 2)
            for _ in range(num_blocks)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(n_qubits) for _ in range(num_blocks)
        ])

        # ── Per-block learnable affine on VQC output (optional) ──────────
        if use_delta:
            self.deltas1 = nn.ParameterList([
                nn.Parameter(torch.ones(n_qubits)) for _ in range(num_blocks)
            ])
            self.deltas2 = nn.ParameterList([
                nn.Parameter(torch.zeros(n_qubits)) for _ in range(num_blocks)
            ])

        # ── Data re-uploading: trainable input/output scaling (Skolik) ───
        if use_reupload:
            # lambda_scales[b]: (n_layers, n_qubits) — per-layer per-qubit input scale
            self.lambda_scales = nn.ParameterList([
                nn.Parameter(torch.ones(n_layers, n_qubits)) for _ in range(num_blocks)
            ])
            # output_scales[b]: (n_qubits,) — per-qubit output scale
            self.output_scales = nn.ParameterList([
                nn.Parameter(torch.ones(n_qubits)) for _ in range(num_blocks)
            ])

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

    def to(self, *args, **kwargs):
        """Keep vqc_blocks on CPU regardless of device transfer."""
        super().to(*args, **kwargs)
        self.vqc_blocks.cpu()
        return self

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

        t_emb = self.time_mlp(t)                           # (B, time_dim)
        c_emb = self.cond_mlp(c)                           # (B, cond_dim)
        cond  = torch.cat([t_emb, c_emb], dim=-1)         # (B, cond_embed_dim)

        x = z_t
        for i, (vqc, c2wb, norm) in enumerate(zip(self.vqc_blocks, self.cond2wb, self.norms)):
            if self.use_reupload:
                # lambda_scales[i]: (n_layers, n_qubits); x: (B, n_qubits)
                # scaled: (B, n_layers, n_qubits) → flatten to (B, n_layers * n_qubits)
                scaled = (self.lambda_scales[i] * x.unsqueeze(1))  # broadcast over batch
                inp = scaled.reshape(x.shape[0], -1)                # (B, n_layers * n_qubits)
                q_out = vqc(inp.cpu().double()).to(x.device).float()
                q_out = self.output_scales[i] * norm(q_out)         # trainable output scale
            else:
                q_out = vqc(x.cpu().double()).to(x.device).float()  # VQC on CPU (statevector), move back
                if self.use_delta:
                    d1, d2 = self.deltas1[i], self.deltas2[i]
                    q_out = d1 * norm(q_out) + d2              # per-dim learnable affine
                else:
                    q_out = norm(q_out)
            w, b  = c2wb(cond).chunk(2, dim=-1)           # (B, D) each
            x     = x + w * q_out + b                     # residual + AdaIN

        return x                                           # ε̂ (B, latent_dim)


def _make_qubit_cond_vqc_layer(
    latent_dim: int,
    n_layers: int,
    device_type: str,
) -> "qml.qnn.TorchLayer":
    """Factory for QubitCondVQCDenoiser blocks.

    Total qubits = latent_dim + 2.
    Inputs: cat([x(D), t(1), sigmoid(c)(1)]) — all scaled by π.
    Measures: qubits 0..D-1 only (conditioning qubits D, D+1 are ancilla).
    """
    _pi = math.pi
    n_total = latent_dim + 2
    dev = qml.device(device_type, wires=n_total)
    params_per_layer = n_total * 3 + n_total * 3
    weight_shapes = {"theta": (params_per_layer * n_layers,)}

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def vqc_circuit(inputs, theta):
        # inputs: (D+2,) = [z_angles(D), t_angle(1), c_angle(1)]
        qml.AngleEmbedding(inputs * _pi, wires=range(n_total), rotation="Y")
        param_idx = 0
        for _ in range(n_layers):
            for i in range(n_total):
                qml.RZ(theta[param_idx],     wires=i)
                qml.RY(theta[param_idx + 1], wires=i)
                qml.RZ(theta[param_idx + 2], wires=i)
                param_idx += 3
            for p in range(n_total):
                nxt = (p + 1) % n_total
                qml.RZ(-_pi / 2,             wires=nxt)
                qml.CNOT(wires=[nxt, p])
                qml.RZ(theta[param_idx],     wires=p)
                qml.RY(theta[param_idx + 1], wires=nxt)
                qml.CNOT(wires=[p, nxt])
                qml.RY(theta[param_idx + 2], wires=nxt)
                param_idx += 3
        # measure only latent qubits (0..D-1), not conditioning qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(latent_dim)]

    return qml.qnn.TorchLayer(vqc_circuit, weight_shapes)


class QubitCondVQCDenoiser(nn.Module):
    """
    VQC denoiser with quantum-native conditioning via dedicated qubits.

    Architecture per block:
        inputs = cat([x(D), t(1), sigmoid(c)(1)])   ← (D+2,)
        AngleEmbedding(π * inputs) on D+2 qubits
        RZY + ring CNOT × n_layers  (t,c qubits entangle with latent qubits)
        measure qubits 0..D-1  →  q_out (B, D)
        x = x + q_out                               ← residual

    t is already in [0,1]; c is z-scored so sigmoid maps it to (0,1).
    Repeat for num_blocks blocks — no classical CondInj needed.
    """

    NULL_COND = 0.0

    def __init__(
        self,
        latent_dim:  int = 8,
        n_layers:    int = 2,
        num_blocks:  int = 6,
        time_dim:    int = 32,   # noqa: kept for interface compatibility with other denoisers
        cond_dim:    int = 32,   # noqa: kept for interface compatibility with other denoisers
        device_type: str = "default.qubit",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_layers   = int(n_layers)
        self.num_blocks = int(num_blocks)

        self.vqc_blocks = nn.ModuleList([
            _make_qubit_cond_vqc_layer(latent_dim, int(n_layers), device_type)
            for _ in range(num_blocks)
        ])

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

    def to(self, *args, **kwargs):
        """Keep vqc_blocks on CPU regardless of device transfer."""
        super().to(*args, **kwargs)
        self.vqc_blocks.cpu()
        return self

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

        # t ∈ [0,1] already; sigmoid(c) maps z-scored pIC50 → (0,1)
        t_enc = t                                          # (B, 1)
        c_enc = torch.sigmoid(c)                          # (B, 1)

        x = z_t
        for vqc in self.vqc_blocks:
            # cat latent + conditioning angles → (B, D+2)
            inp = torch.cat([x, t_enc, c_enc], dim=-1)   # (B, D+2)
            q_out = vqc(inp.cpu().double()).to(inp.device).float()  # VQC on CPU
            x = x + q_out                                # residual

        return x                                          # ε̂ (B, latent_dim)
