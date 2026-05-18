"""
VQC denoisers for latent CFG diffusion.

Classes:
    AngleVQCDenoiser    — num_blocks × [AngleVQC + CondInj] with tanh input bounding
                          Supports angle_reupload (AngleEmbedding) and zz_reupload (ZZ-FeatureMap)
    QubitCondVQCDenoiser — quantum-native conditioning via dedicated ancilla qubits
"""
from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn

def _make_reupload_vqc_layer(
    n_qubits: int,
    n_layers: int,
    device_type: str,
    initial_cnot: bool = False,
) -> "qml.qnn.TorchLayer":
    """Factory for data re-uploading VQC blocks (Skolik architecture).

    inputs shape: (n_layers * n_qubits,) — pre-scaled by lambda_scales outside the qnode.
    At each layer l, inputs[l*n_qubits:(l+1)*n_qubits] are re-encoded via AngleEmbedding.
    initial_cnot: if True, a fixed CNOT ring is inserted once after the first encoding.
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
            x_l = inputs[..., l * n_qubits : (l + 1) * n_qubits]
            qml.AngleEmbedding(x_l * _pi, wires=range(n_qubits), rotation="Y")
            # Fixed CNOT ring once after first encoding
            if initial_cnot and l == 0:
                for j in range(n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % n_qubits])
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


def _make_zz_reupload_vqc_layer(
    n_qubits: int,
    n_layers: int,
    device_type: str,
    initial_cnot: bool = False,
) -> "qml.qnn.TorchLayer":
    """Factory for ZZ-FeatureMap data re-uploading VQC blocks.

    Replaces AngleEmbedding with ZZ-FeatureMap per re-uploading layer:
      H on all qubits → RZ(2·x_i·π) per qubit → CNOT-RZ(2·(π-x_i·π)(π-x_j·π))-CNOT per pair
    inputs shape: (n_layers * n_qubits,) — pre-scaled by lambda_scales outside the qnode.
    Ansatz (variational rotations + ring entanglement) is identical to angle_reupload.
    """
    _pi = math.pi
    dev = qml.device(device_type, wires=n_qubits)
    params_per_layer = n_qubits * 3 + n_qubits * 3
    weight_shapes = {"theta": (params_per_layer * n_layers,)}

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def vqc_circuit(inputs, theta):
        param_idx = 0
        for l in range(n_layers):
            x_l = inputs[..., l * n_qubits : (l + 1) * n_qubits]
            # ZZ-FeatureMap
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.RZ(2.0 * x_l[..., i] * _pi, wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(2.0 * (_pi - x_l[..., i] * _pi) * (_pi - x_l[..., i + 1] * _pi), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            if initial_cnot and l == 0:
                for j in range(n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % n_qubits])
            # Variational rotations (same as angle_reupload)
            for i in range(n_qubits):
                qml.RZ(theta[param_idx],     wires=i)
                qml.RY(theta[param_idx + 1], wires=i)
                qml.RZ(theta[param_idx + 2], wires=i)
                param_idx += 3
            # Ring entanglement (same as angle_reupload)
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
    initial_cnot: bool = False,
) -> "qml.qnn.TorchLayer":
    """Factory to create an independent TorchLayer for each VQC block.

    initial_cnot: if True, a fixed CNOT ring is inserted after AngleEmbedding.
    """
    _pi = math.pi
    dev = qml.device(device_type, wires=n_qubits)
    params_per_layer = n_qubits * 3 + n_qubits * 3  # rot + entangle
    weight_shapes = {"theta": (params_per_layer * n_layers,)}

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def vqc_circuit(inputs, theta):
        qml.AngleEmbedding(inputs * _pi, wires=range(n_qubits), rotation="Y")
        if initial_cnot:
            for j in range(n_qubits):
                qml.CNOT(wires=[j, (j + 1) % n_qubits])
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
        use_delta:     bool = False,  # per-block learnable affine on VQC output (δ1·norm(q)+δ2)
        use_reupload:  bool = False,  # data re-uploading: encode x at start of every layer
        initial_cnot:  bool = False,  # fixed CNOT ring after initial AngleEmbedding
        full_encoding: bool = False,  # full matrix W@x encoding instead of diagonal λ*x
        use_zz:        bool = False,  # ZZ-FeatureMap instead of AngleEmbedding (requires use_reupload)
    ):
        super().__init__()
        self.latent_dim    = int(latent_dim)
        self.n_layers      = int(n_layers)
        self.num_blocks    = int(num_blocks)
        self.time_dim      = int(time_dim)
        self.cond_dim      = int(cond_dim)
        self.use_delta     = bool(use_delta)
        self.use_reupload  = bool(use_reupload)
        self.initial_cnot  = bool(initial_cnot)
        self.full_encoding = bool(full_encoding) and bool(use_reupload)
        self.use_zz        = bool(use_zz) and bool(use_reupload)

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
        if use_reupload and self.use_zz:
            self.vqc_blocks = nn.ModuleList([
                _make_zz_reupload_vqc_layer(n_qubits, int(n_layers), device_type, initial_cnot=initial_cnot)
                for _ in range(num_blocks)
            ])
        elif use_reupload:
            self.vqc_blocks = nn.ModuleList([
                _make_reupload_vqc_layer(n_qubits, int(n_layers), device_type, initial_cnot=initial_cnot)
                for _ in range(num_blocks)
            ])
        else:
            self.vqc_blocks = nn.ModuleList([
                _make_angle_vqc_layer(n_qubits, int(n_layers), device_type, initial_cnot=initial_cnot)
                for _ in range(num_blocks)
            ])

        # ── CondInj per block: cond → (w, b) each of size D ──────────────
        # Matches MLP _CondInjectionLayer: Linear→SiLU→Linear hidden projection.
        self.cond2wb = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cond_embed_dim, cond_embed_dim),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, n_qubits * 2),
            ) for _ in range(num_blocks)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(n_qubits) for _ in range(num_blocks)
        ])
        # Matches MLP _CondInjectionLayer f2: post-AdaIN projection.
        self.f2 = nn.ModuleList([
            nn.Linear(n_qubits, n_qubits) for _ in range(num_blocks)
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
            if self.full_encoding:
                # weight_matrices[b]: (n_layers, n_qubits, n_qubits) — full W@x encoding
                # init to identity so W@x = x at start (equivalent to λ=1 diagonal)
                self.weight_matrices = nn.ParameterList([
                    nn.Parameter(
                        torch.eye(n_qubits).unsqueeze(0).repeat(n_layers, 1, 1)
                    ) for _ in range(num_blocks)
                ])
            else:
                # lambda_scales[b]: (n_layers, n_qubits) — diagonal λ*x encoding
                self.lambda_scales = nn.ParameterList([
                    nn.Parameter(torch.ones(n_layers, n_qubits)) for _ in range(num_blocks)
                ])
            # input_biases[b]: (n_layers, n_qubits) — per-layer per-qubit bias (UQC: ω·x + α)
            self.input_biases = nn.ParameterList([
                nn.Parameter(torch.zeros(n_layers, n_qubits)) for _ in range(num_blocks)
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
        for i, (vqc, c2wb, norm, f2) in enumerate(zip(self.vqc_blocks, self.cond2wb, self.norms, self.f2)):
            x_enc = torch.tanh(x)
            if self.use_reupload:
                if self.full_encoding:
                    scaled = torch.einsum('lod,bd->blo', self.weight_matrices[i], x_enc) \
                             + self.input_biases[i]
                else:
                    scaled = self.lambda_scales[i] * x_enc.unsqueeze(1) + self.input_biases[i]
                inp = scaled.reshape(x.shape[0], -1)                # (B, n_layers * n_qubits)
                q_out = vqc(inp.cpu().double()).to(x.device).float()
                q_out = self.output_scales[i] * norm(q_out)         # trainable output scale
            else:
                q_out = vqc(x_enc.cpu().double()).to(x.device).float()
                if self.use_delta:
                    d1, d2 = self.deltas1[i], self.deltas2[i]
                    q_out = d1 * norm(q_out) + d2
                else:
                    q_out = norm(q_out)
            w, b = c2wb(cond).chunk(2, dim=-1)            # (B, D) each
            h    = w * q_out + b                          # AdaIN scale+shift
            h    = f2(torch.nn.functional.gelu(h))        # post-AdaIN projection (matches MLP f2)
            x    = x + h                                  # residual

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
