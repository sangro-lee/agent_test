import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import numpy as standard_np


class VQCPropertyPredictor_torch(nn.Module):
    """
    """

    def __init__(self, latent_dim=256, n_qubits=8, n_layers=1, device_type="default.qubit"):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_type = device_type

        self.required_dim = 2 ** self.n_qubits  # e.g., 8 qubits → 256

        if latent_dim == self.required_dim:
            self.dim_adapter = nn.Identity()
        else:
            self.dim_adapter = nn.Linear(latent_dim, self.required_dim) #for check dim variation

        if device_type == "lightning.gpu":
            self.dev = qml.device("lightning.gpu", wires=self.n_qubits)
        else:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

        weight_shapes = self._get_weight_shapes()
        circuit = self._create_circuit()

        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.delta_1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.delta_2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _get_weight_shapes(self):
        params_per_layer = (self.n_qubits * 3) + ((self.n_qubits - 1) * 3)
        n_params = params_per_layer * self.n_layers
        return {"theta": (n_params,)}

    def _create_circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, theta):
            # Input embedding
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=False)

            param_idx = 0  # Parameter index tracker


            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            for layer in range(self.n_layers):
                # Part 1: Rotation layer (24 params)
                for i in range(self.n_qubits):
                    qml.RZ(theta[param_idx], wires=i)
                    qml.RY(theta[param_idx + 1], wires=i)
                    qml.RZ(theta[param_idx + 2], wires=i)
                    param_idx += 3
                # param_idx now at layer*45 + 24

                # Part 2: Entanglement layer (dynamic based on n_qubits)
                for pattern in range(self.n_qubits - 1):
                    target_qubit = pattern + 1
                    control_qubit = pattern

                    qml.RZ(-torch.pi / 2, wires=target_qubit)
                    qml.CNOT(wires=[target_qubit, control_qubit])
                    qml.RZ(theta[param_idx], wires=control_qubit)
                    qml.RY(theta[param_idx + 1], wires=target_qubit)
                    qml.CNOT(wires=[control_qubit, target_qubit])
                    qml.RY(theta[param_idx + 2], wires=target_qubit)
                    param_idx += 3

            return qml.expval(qml.PauliZ(self.n_qubits - 1))

        return circuit

    def _prepare_input(self, latent_vector):
        latent_vector = self.dim_adapter(latent_vector)
        return latent_vector

    def forward(self, latent_vector):

        prepared_input = self._prepare_input(latent_vector).double()

        prepared_input = torch.nan_to_num(prepared_input, nan=0.0, posinf=0.0, neginf=0.0)

        norms = prepared_input.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        prepared_input = prepared_input / norms
        """
        with torch.no_grad():
            norms_after = prepared_input.pow(2).sum(dim=-1).sqrt()
            print(
                "[VQC] AFTER norms: min =",
                norms_after.min().item(),
                "max =",
                norms_after.max().item(),
            )
            print(
                "[VQC] AFTER any_nan:",
                torch.isnan(prepared_input).any().item(),
                "any_inf:",
                torch.isinf(prepared_input).any().item(),
            )
        """
        if torch.isnan(prepared_input).any() or torch.isinf(prepared_input).any():
            raise RuntimeError("[VQC] prepared_input has NaN/Inf before AmplitudeEmbedding")

        measurements = self.vqc(prepared_input)  # [B]
        measurements_f32 = measurements.float()

        predictions = self.delta_1 * measurements_f32 + self.delta_2
        predictions = predictions.unsqueeze(-1)  # [B, 1]

        return predictions

    def get_circuit_info(self):
        params_per_layer = (self.n_qubits * 3) + ((self.n_qubits - 1) * 3)
        n_params = params_per_layer * self.n_layers

        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_params': n_params,
            'latent_dim': self.latent_dim,
            'required_dim': self.required_dim,
            'device': str(self.dev),
            'total_trainable_params': n_params + 2,  # θ parameters + δ₁, δ₂
            'uses_torchlayer': True
        }


def create_vqc_property_predictor(predictor_type="vqc", latent_dim=256, device_type="lightning.qubit", **kwargs):

    if predictor_type == "vqc_torch":
        return VQCPropertyPredictor_torch(latent_dim=latent_dim, device_type=device_type, **kwargs)
    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}. Choose from 'vqc', 'optimized', 'batched', 'vmap', 'hybrid'")


class VQCEncoderHead(nn.Module):
    """
    VQC encoder head: latent_dim → AmplitudeEmbedding(n_qubits) → n_qubits-dim output.
    All qubits measured: [expval(Z(i)) for i in range(n_qubits)].

    Replaces the final projection layer of GCN/MLP backbone to produce
    a quantum-compressed n_qubits-dim latent (e.g., 256 → 8).
    """

    def __init__(self, latent_dim: int = 256, n_qubits: int = 8, n_layers: int = 1,
                 device_type: str = "default.qubit"):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        required_dim = 2 ** n_qubits
        if latent_dim == required_dim:
            self.dim_adapter = nn.Identity()
        else:
            self.dim_adapter = nn.Linear(latent_dim, required_dim)

        dev = qml.device(device_type, wires=n_qubits)

        params_per_layer = (n_qubits * 3) + ((n_qubits - 1) * 3)
        weight_shapes = {"theta": (params_per_layer * n_layers,)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, theta):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=False)

            param_idx = 0
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            for _ in range(n_layers):
                for i in range(n_qubits):
                    qml.RZ(theta[param_idx], wires=i)
                    qml.RY(theta[param_idx + 1], wires=i)
                    qml.RZ(theta[param_idx + 2], wires=i)
                    param_idx += 3
                for p in range(n_qubits - 1):
                    qml.RZ(-torch.pi / 2, wires=p + 1)
                    qml.CNOT(wires=[p + 1, p])
                    qml.RZ(theta[param_idx], wires=p)
                    qml.RY(theta[param_idx + 1], wires=p + 1)
                    qml.CNOT(wires=[p, p + 1])
                    qml.RY(theta[param_idx + 2], wires=p + 1)
                    param_idx += 3

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Per-qubit affine post-processing: delta1 * x + delta2
        # To disable, comment out the two lines below and the corresponding line in forward()
        self.delta1 = nn.Parameter(torch.ones(n_qubits))
        self.delta2 = nn.Parameter(torch.zeros(n_qubits))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            out: (B, n_qubits)
        """
        x = self.dim_adapter(z).double()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        norms = x.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        x = x / norms
        out = self.vqc(x).float()   # (B, n_qubits) in [-1, 1]^n_qubits
        out = self.delta1 * out + self.delta2  # per-qubit affine; comment out to disable
        return out


class VQCConditionalDenoiser(nn.Module):
    """
    VQC-based CFG denoiser for n_qubits-dim latent space.

    Architecture (data re-uploading):
        inputs = [z_t(n_qubits) | t(1) | c(1)]  →  TorchLayer
        Circuit per layer:
            AngleEmbedding(z_part) → RY(weights[l,i]) → CNOT chain
            → RY(t * scale_t[i])  (data re-uploading for t)
            → RY(c * scale_c[i])  (data re-uploading for c)
        Output: [expval(Z(i)) for i in range(n_qubits)]  ← ε̂

    Compatible interface with ConditionalDenoisingMLP
    (same set_normalization / forward signature, NULL_COND sentinel).
    """

    NULL_COND = 0.0

    def __init__(self, latent_dim: int = 8, n_qubits: int = 8, n_layers: int = 2,
                 device_type: str = "default.qubit"):
        super().__init__()
        assert latent_dim == n_qubits, (
            f"VQCConditionalDenoiser requires latent_dim == n_qubits, "
            f"got latent_dim={latent_dim} vs n_qubits={n_qubits}"
        )
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Normalization buffers (same interface as ConditionalDenoisingMLP)
        self.register_buffer("z_mean", torch.zeros(n_qubits))
        self.register_buffer("z_std", torch.ones(n_qubits))
        self.register_buffer("c_mean", torch.zeros(1))
        self.register_buffer("c_std", torch.ones(1))

        dev = qml.device(device_type, wires=n_qubits)

        weight_shapes = {
            "weights": (n_layers, n_qubits),   # main rotation params per layer
            "scale_t": (n_qubits,),             # per-qubit scale for timestep t
            "scale_c": (n_qubits,),             # per-qubit scale for condition c
        }

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights, scale_t, scale_c):
            # inputs: (n_qubits + 2,) per sample
            #   [:n_qubits] = z_t (normalized latent)
            #   [n_qubits]  = t   (normalized timestep in [0,1])
            #   [n_qubits+1]= c   (normalized pIC50; 0.0 = NULL_COND)
            z_part = inputs[:n_qubits]
            t_val = inputs[n_qubits]
            c_val = inputs[n_qubits + 1]

            # Encode z_t via AngleEmbedding (RY gates, one per qubit)
            qml.AngleEmbedding(z_part, wires=range(n_qubits), rotation="Y")

            for layer in range(n_layers):
                # Parameterized rotation layer
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Data re-uploading: inject t and c as per-qubit rotations
                for i in range(n_qubits):
                    qml.RY(t_val * scale_t[i], wires=i)
                    qml.RY(c_val * scale_c[i], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Initialize scale params to small values so early training is stable
        torch.nn.init.constant_(self.vqc.scale_t, 0.1)
        torch.nn.init.constant_(self.vqc.scale_c, 0.1)

    def set_normalization(self, z_mean: torch.Tensor, z_std: torch.Tensor,
                          c_mean: torch.Tensor = None, c_std: torch.Tensor = None):
        self.z_mean.copy_(z_mean.detach())
        self.z_std.copy_(z_std.detach())
        if c_mean is not None:
            self.c_mean.copy_(c_mean.detach().view(1))
        if c_std is not None:
            self.c_std.copy_(c_std.detach().view(1))

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: (B, n_qubits)
            t:   (B,) or (B, 1), normalized to [0, 1]
            c:   (B, 1), normalized pIC50. Use NULL_COND (0.0) for unconditional.
        Returns:
            eps_pred: (B, n_qubits)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if c.dim() == 1:
            c = c.unsqueeze(1)

        # Concatenate [z_t | t | c] → (B, n_qubits + 2), convert to double for PennyLane
        inputs = torch.cat([z_t, t, c], dim=-1).double()
        eps = self.vqc(inputs)   # (B, n_qubits)
        return eps.float()

