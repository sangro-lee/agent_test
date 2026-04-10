from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.diffusion import ConditionalDenoisingMLP, DenoisingMLP, NoiseScheduler
from src.models.vqc_diffusion import AngleVQCDenoiser, HybridUNetDenoiser, QubitCondVQCDenoiser
from src.models.vqc_module import VQCConditionalDenoiser


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _predict_from_latent(model, z: torch.Tensor) -> torch.Tensor:
    """Extract the pIC50 prediction head from the trained encoder model."""
    if hasattr(model, "out_layer"):
        return model.out_layer(z).view(-1)
    if hasattr(model, "reg_head"):
        return model.reg_head(z).view(-1)
    raise ValueError("Model does not expose a latent->prediction head (out_layer or reg_head).")


# ---------------------------------------------------------------------------
# Legacy gradient-ascent optimizer (kept as comparison baseline)
# ---------------------------------------------------------------------------

def optimize_latent_vector(model, z_init, n_steps: int, lr: float):
    """
    Simple gradient ascent on pIC50 in latent space.
    Baseline comparison for diffusion-based methods.
    """
    device = next(model.parameters()).device
    z = torch.tensor(z_init, dtype=torch.float32, device=device).view(1, -1)
    z = torch.nn.Parameter(z)

    optimizer = torch.optim.Adam([z], lr=lr)
    model.eval()

    for _ in range(int(n_steps)):
        optimizer.zero_grad()
        pred = _predict_from_latent(model, z)
        loss = -pred.mean()  # gradient ascent on predicted activity
        loss.backward()
        optimizer.step()

    return z.detach().cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# Nearest-neighbor retrieval
# ---------------------------------------------------------------------------

def retrieve_nearest(
    z_opt,
    z_train,
    smiles_train: List[str],
    top_k: int,
) -> List[Tuple[str, float]]:
    z_opt = np.asarray(z_opt, dtype=float).reshape(1, -1)
    z_train = np.asarray(z_train, dtype=float)

    if z_train.ndim != 2 or z_train.shape[0] != len(smiles_train):
        raise ValueError("z_train shape must be (N, D) and match smiles_train length.")

    z_opt_norm = z_opt / (np.linalg.norm(z_opt, axis=1, keepdims=True) + 1e-12)
    z_train_norm = z_train / (np.linalg.norm(z_train, axis=1, keepdims=True) + 1e-12)
    sims = (z_train_norm @ z_opt_norm.T).reshape(-1)

    order = np.argsort(-sims)[: int(top_k)]
    return [(smiles_train[i], float(sims[i])) for i in order]


# ---------------------------------------------------------------------------
# Unconditional diffusion training  (used by gradient-guidance sampling)
# ---------------------------------------------------------------------------

def train_diffusion(
    z_train,
    latent_dim: int = 128,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 2e-4,
    T: int = 1000,
    time_dim: int = 64,
    hidden_dim: int = 512,
    device: str = "cpu",
):
    """
    Train an UNCONDITIONAL DenoisingMLP on z_train latents.
    Used together with gradient guidance at inference.

    Returns:
        denoiser: DenoisingMLP (with normalization stats stored as buffers)
        stats:    (mean_np, std_np) for external use
    """
    device_t = torch.device(device)
    z_train = np.asarray(z_train, dtype=np.float32)
    if z_train.ndim != 2:
        raise ValueError("z_train must be 2D array of shape (N, D).")

    latent_dim = int(latent_dim)
    assert z_train.shape[1] == latent_dim, f"latent_dim mismatch: expected {latent_dim}, got {z_train.shape[1]}"

    mean_np = z_train.mean(axis=0)
    std_np = z_train.std(axis=0)
    std_np = np.where(std_np < 1e-6, 1.0, std_np)

    z_norm = (z_train - mean_np) / std_np
    loader = DataLoader(
        TensorDataset(torch.tensor(z_norm, dtype=torch.float32)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    denoiser = DenoisingMLP(latent_dim=latent_dim, time_dim=int(time_dim), hidden_dim=int(hidden_dim)).to(device_t)
    denoiser.set_normalization(
        torch.tensor(mean_np, dtype=torch.float32, device=device_t),
        torch.tensor(std_np, dtype=torch.float32, device=device_t),
    )

    scheduler = NoiseScheduler(T=int(T), device=device_t)
    opt = torch.optim.Adam(denoiser.parameters(), lr=float(lr))

    denoiser.train()
    for epoch in range(int(epochs)):
        for (z0_batch,) in loader:
            z0_batch = z0_batch.to(device_t)
            t_idx = torch.randint(0, int(T), (z0_batch.size(0),), device=device_t)
            t_norm = (t_idx.float().unsqueeze(1) + 1.0) / float(T)
            z_t, eps = scheduler.add_noise(z0_batch, t_idx)
            eps_pred = denoiser(z_t, t_norm)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"  [diffusion uncond] epoch {epoch+1}/{epochs} loss={loss.item():.4f}")

    denoiser.eval()
    return denoiser, (mean_np.astype(np.float32), std_np.astype(np.float32))


# ---------------------------------------------------------------------------
# CFG (Classifier-Free Guidance) diffusion training
# ---------------------------------------------------------------------------

def train_diffusion_cfg(
    z_train,
    y_train,                    # pIC50 values, shape (N,)
    z_val=None,                 # optional val latents
    y_val=None,                 # optional val pIC50 values
    latent_dim: int = 128,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 2e-4,
    T: int = 1000,
    time_dim: int = 128,
    cond_dim: int = 128,
    hidden_dim: int = 512,
    p_uncond: float = 0.15,     # probability of dropping condition during training
    model_type: str = "mlp",    # "mlp" or "vqc"
    denoiser_type: str = "mlp", # "mlp" | "unet" | "unet_vqc" | "vqc_angle"
    unet_dims: list | None = None,
    n_layers: int = 2,
    num_blocks: int = 6,        # number of VQC+CondInj blocks for vqc_angle
    initial_cnot:  bool = False,  # fixed CNOT ring after AngleEmbedding
    full_encoding: bool = False,  # full W@x matrix encoding (reupload only)
    device: str = "cpu",
    best_ckpt_path=None,            # Path | None — save immediately on best val loss
    ckpt_meta: dict | None = None,  # extra fields merged into checkpoint
    history_path=None,              # Path | None — write loss CSV after every epoch
):
    """
    Train a CONDITIONAL DenoisingMLP with classifier-free guidance.
    During training, condition (pIC50) is randomly dropped with p_uncond,
    replaced by NULL_COND (0.0) so the model learns both cond & uncond distributions.

    At inference:
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
    where w = guidance_scale.

    Returns:
        denoiser: ConditionalDenoisingMLP
        stats:    (z_mean, z_std, c_mean, c_std)
    """
    device_t = torch.device(device)
    z_train = np.asarray(z_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    assert z_train.shape[0] == y_train.shape[0], "z_train and y_train must have same length"

    latent_dim = int(latent_dim)

    # Normalize latents
    z_mean = z_train.mean(axis=0)
    z_std = z_train.std(axis=0)
    z_std = np.where(z_std < 1e-6, 1.0, z_std)
    z_norm = (z_train - z_mean) / z_std

    # Normalize condition (pIC50)
    c_mean = float(y_train.mean())
    c_std = float(y_train.std())
    c_std = max(c_std, 1e-6)
    c_norm = (y_train - c_mean) / c_std  # shape (N,)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(z_norm, dtype=torch.float32),
            torch.tensor(c_norm, dtype=torch.float32),
        ),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    val_loader = None
    if z_val is not None and y_val is not None:
        z_val = np.asarray(z_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
        z_val_norm = (z_val - z_mean) / z_std
        c_val_norm = (y_val - c_mean) / c_std
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(z_val_norm, dtype=torch.float32),
                torch.tensor(c_val_norm, dtype=torch.float32),
            ),
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
        )

    if model_type == "vqc":
        denoiser = VQCConditionalDenoiser(
            latent_dim=latent_dim,
            n_qubits=latent_dim,
        ).to(device_t)
    elif denoiser_type in ("vqc_angle", "vqc_angle_delta", "vqc_angle_reupload"):
        denoiser = AngleVQCDenoiser(
            latent_dim=latent_dim,
            n_layers=int(n_layers),
            num_blocks=int(num_blocks),
            time_dim=int(time_dim),
            cond_dim=int(cond_dim),
            use_delta=(denoiser_type == "vqc_angle_delta"),
            use_reupload=(denoiser_type == "vqc_angle_reupload"),
            initial_cnot=initial_cnot,
            full_encoding=full_encoding,
        ).to(device_t)
    elif denoiser_type == "vqc_qubit_cond":
        denoiser = QubitCondVQCDenoiser(
            latent_dim=latent_dim,
            n_layers=int(n_layers),
            num_blocks=int(num_blocks),
        ).to(device_t)
    elif denoiser_type in ("unet", "unet_vqc"):
        denoiser = HybridUNetDenoiser(
            latent_dim=latent_dim,
            unet_dims=unet_dims,
            n_layers=int(n_layers),
            time_dim=int(time_dim),
            cond_dim=int(cond_dim),
            use_vqc=(denoiser_type == "unet_vqc"),
        ).to(device_t)
    elif denoiser_type == "mlp_ortho":
        denoiser = ConditionalDenoisingMLP(
            latent_dim=latent_dim,
            time_dim=int(time_dim),
            cond_dim=int(cond_dim),
            hidden_dim=int(hidden_dim),
            use_orthogonal=True,
        ).to(device_t)
    else:
        denoiser = ConditionalDenoisingMLP(
            latent_dim=latent_dim,
            time_dim=int(time_dim),
            cond_dim=int(cond_dim),
            hidden_dim=int(hidden_dim),
        ).to(device_t)
    denoiser.set_normalization(
        torch.tensor(z_mean, dtype=torch.float32, device=device_t),
        torch.tensor(z_std, dtype=torch.float32, device=device_t),
        torch.tensor([c_mean], dtype=torch.float32, device=device_t),
        torch.tensor([c_std], dtype=torch.float32, device=device_t),
    )

    scheduler = NoiseScheduler(T=int(T), device=device_t)
    opt = torch.optim.Adam(denoiser.parameters(), lr=float(lr))

    null_cond = denoiser.NULL_COND

    import copy
    best_loss = float("inf")
    best_state_dict = None
    history = []  # list of {"epoch", "train_loss", "val_loss"}

    # Check if model has VQC parameters for gradient monitoring
    _has_vqc = any("vqc" in n for n, _ in denoiser.named_parameters())

    denoiser.train()
    for epoch in range(int(epochs)):
        epoch_loss = 0.0
        n_batches = 0
        vqc_grad_sum = 0.0
        cls_grad_sum = 0.0
        for z0_batch, c_batch in loader:
            z0_batch = z0_batch.to(device_t)
            c_batch = c_batch.to(device_t).unsqueeze(1)  # (B, 1)

            # Random condition dropout → replace with null
            drop_mask = torch.rand(c_batch.size(0), 1, device=device_t) < p_uncond
            c_input = c_batch.masked_fill(drop_mask, null_cond)

            t_idx = torch.randint(0, int(T), (z0_batch.size(0),), device=device_t)
            t_norm = (t_idx.float().unsqueeze(1) + 1.0) / float(T)

            z_t, eps = scheduler.add_noise(z0_batch, t_idx)
            eps_pred = denoiser(z_t, t_norm, c_input)

            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            if _has_vqc:
                vqc_norms, cls_norms = [], []
                for n, p in denoiser.named_parameters():
                    if p.grad is not None:
                        norm = p.grad.norm().item()
                        if "vqc" in n:
                            vqc_norms.append(norm)
                        else:
                            cls_norms.append(norm)
                if vqc_norms:
                    vqc_grad_sum += sum(vqc_norms) / len(vqc_norms)
                if cls_norms:
                    cls_grad_sum += sum(cls_norms) / len(cls_norms)

            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # Val loss
        val_loss = None
        if val_loader is not None:
            denoiser.eval()
            val_epoch_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for z0_batch, c_batch in val_loader:
                    z0_batch = z0_batch.to(device_t)
                    c_batch = c_batch.to(device_t).unsqueeze(1)
                    t_idx = torch.randint(0, int(T), (z0_batch.size(0),), device=device_t)
                    t_norm = (t_idx.float().unsqueeze(1) + 1.0) / float(T)
                    z_t, eps = scheduler.add_noise(z0_batch, t_idx)
                    eps_pred = denoiser(z_t, t_norm, c_batch)
                    val_epoch_loss += F.mse_loss(eps_pred, eps).item()
                    val_batches += 1
            val_loss = val_epoch_loss / max(val_batches, 1)
            denoiser.train()

        monitor_loss = val_loss if val_loss is not None else train_loss
        improved = monitor_loss < best_loss
        if improved:
            best_loss = monitor_loss
            best_state_dict = copy.deepcopy(denoiser.state_dict())
            if best_ckpt_path is not None:
                import torch as _torch
                from pathlib import Path as _Path
                _Path(best_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                _ckpt = {"state_dict": best_state_dict,
                         "z_mean": z_mean, "z_std": z_std,
                         "c_mean": c_mean, "c_std": c_std}
                if ckpt_meta:
                    _ckpt.update(ckpt_meta)
                _torch.save(_ckpt, best_ckpt_path)

        row = {"epoch": epoch + 1, "train_loss": train_loss}
        if val_loss is not None:
            row["val_loss"] = val_loss
        if _has_vqc:
            row["vqc_grad_norm"]      = vqc_grad_sum / max(n_batches, 1)
            row["classical_grad_norm"] = cls_grad_sum / max(n_batches, 1)
        history.append(row)

        if history_path is not None:
            import csv as _csv
            from pathlib import Path as _Path
            _hp = _Path(history_path)
            _hp.parent.mkdir(parents=True, exist_ok=True)
            with open(_hp, "w", newline="") as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(history[0].keys()))
                _w.writeheader()
                _w.writerows(history)

        msg = f"  [diffusion cfg] epoch {epoch+1}/{epochs}  loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f"  val={val_loss:.4f}"
        msg += f"  best={best_loss:.4f}"
        if improved:
            msg += "  *"
        if _has_vqc:
            msg += (f"  vqc_grad={row['vqc_grad_norm']:.2e}"
                    f"  cls_grad={row['classical_grad_norm']:.2e}")
        print(msg)

    denoiser.eval()
    stats = (z_mean.astype(np.float32), z_std.astype(np.float32), c_mean, c_std)
    return denoiser, best_state_dict, stats, history


# ---------------------------------------------------------------------------
# Sampling: Gradient Guidance (classifier guidance on reg_head)
# ---------------------------------------------------------------------------

def sample_gradient_guidance(
    denoiser: DenoisingMLP,
    reg_head: nn.Module,
    latent_dim: int = 128,
    n_samples: int = 500,
    T: int = 1000,
    guidance_scale: float = 3.0,
    sampler: str = "ddim",
    device: str = "cpu",
) -> np.ndarray:
    """
    Sampling with GRADIENT GUIDANCE (classifier guidance variant).
    Uses unconditional denoiser + gradient of reg_head at each step.

    Guidance formula (score-based):
        eps_guided = eps_pred - sqrt(1 - alpha_bar_t) * guidance_scale * ∇_z pIC50(z_t)

    The minus sign ensures we move TOWARD higher pIC50.
    (Larger pIC50 gradient → subtract from eps → decoder moves in gradient direction)
    """
    device_t = torch.device(device)
    denoiser = denoiser.to(device_t).eval()
    reg_head = reg_head.to(device_t).eval()

    for p in denoiser.parameters():
        p.requires_grad_(False)
    for p in reg_head.parameters():
        p.requires_grad_(False)

    scheduler = NoiseScheduler(T=int(T), device=device_t)
    z_mean = denoiser.z_mean.to(device_t)
    z_std = denoiser.z_std.to(device_t)

    z_t = torch.randn(int(n_samples), int(latent_dim), device=device_t)

    for t_int in range(int(T), 0, -1):
        t_norm = torch.full((z_t.size(0), 1), float(t_int) / float(T), device=device_t)

        with torch.no_grad():
            eps_pred = denoiser(z_t, t_norm)

        if float(guidance_scale) > 0.0:
            # Gradient of pIC50 w.r.t. z_t (in normalized space)
            z_req = z_t.detach().clone().requires_grad_(True)
            z_orig = z_req * z_std.unsqueeze(0) + z_mean.unsqueeze(0)
            score = reg_head(z_orig).view(-1).mean()
            grad = torch.autograd.grad(score, z_req)[0].detach()

            # Score-based guidance: subtract scaled gradient from eps
            # (eps ↓ along grad direction → z_{t-1} ↑ along grad direction → pIC50 ↑)
            alpha_bar_t = scheduler.alpha_bars[t_int - 1].to(device_t)
            eps_pred = eps_pred - torch.sqrt(1.0 - alpha_bar_t) * guidance_scale * grad

        if sampler.lower() == "ddpm":
            z_t = scheduler.ddpm_step(z_t, t_int=t_int, eps_pred=eps_pred)
        else:  # ddim
            z_t = scheduler.ddim_step(z_t, t_int=t_int, eps_pred=eps_pred, eta=0.0)

    # Denormalize
    z0 = z_t * z_std.unsqueeze(0) + z_mean.unsqueeze(0)
    return z0.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Sampling: Classifier-Free Guidance (CFG)
# ---------------------------------------------------------------------------

def sample_cfg(
    denoiser: ConditionalDenoisingMLP,
    target_pic50: float,          # target pIC50 to condition on (original scale)
    latent_dim: int = 128,
    n_samples: int = 500,
    T: int = 1000,
    guidance_scale: float = 3.0,  # w in: eps_uncond + w*(eps_cond - eps_uncond)
    sampler: str = "ddim",
    device: str = "cpu",
    traj_every: int = 0,          # >0: save denormalized z every N steps
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Sampling with CLASSIFIER-FREE GUIDANCE (CFG).
    Trains both conditional & unconditional via p_uncond dropout.
    At inference: eps_guided = eps_uncond + w * (eps_cond - eps_uncond)

    w=0  → pure unconditional (sample from learned latent distribution)
    w=1  → pure conditional (follow condition exactly)
    w>1  → amplify condition signal (extrapolate beyond training distribution)

    Args:
        target_pic50: desired pIC50 value in original scale (e.g., 8.0)
        traj_every: if >0, record denormalized z_t every N steps.
                    Returns trajectory array of shape (n_checkpoints, n_samples, latent_dim).

    Returns:
        (z0, trajectory) where trajectory is None when traj_every=0.
    """
    device_t = torch.device(device)
    denoiser = denoiser.to(device_t).eval()

    for p in denoiser.parameters():
        p.requires_grad_(False)

    scheduler = NoiseScheduler(T=int(T), device=device_t)
    z_mean = denoiser.z_mean.to(device_t)
    z_std = denoiser.z_std.to(device_t)
    c_mean = denoiser.c_mean.to(device_t)
    c_std = denoiser.c_std.to(device_t)

    # Normalize target condition
    c_val_norm = (target_pic50 - c_mean.item()) / c_std.item()
    c_cond = torch.full((int(n_samples), 1), c_val_norm, device=device_t)
    c_null = torch.full((int(n_samples), 1), ConditionalDenoisingMLP.NULL_COND, device=device_t)

    z_t = torch.randn(int(n_samples), int(latent_dim), device=device_t)
    traj_frames: list[np.ndarray] = []

    def _denorm(z: torch.Tensor) -> np.ndarray:
        return (z * z_std.unsqueeze(0) + z_mean.unsqueeze(0)).detach().cpu().numpy()

    with torch.no_grad():
        for t_int in range(int(T), 0, -1):
            t_norm = torch.full((z_t.size(0), 1), float(t_int) / float(T), device=device_t)

            eps_cond = denoiser(z_t, t_norm, c_cond)
            eps_uncond = denoiser(z_t, t_norm, c_null)

            # CFG interpolation
            eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            if sampler.lower() == "ddpm":
                z_t = scheduler.ddpm_step(z_t, t_int=t_int, eps_pred=eps_guided)
            else:  # ddim
                z_t = scheduler.ddim_step(z_t, t_int=t_int, eps_pred=eps_guided, eta=0.0)

            if traj_every > 0 and (int(T) - t_int) % traj_every == 0:
                traj_frames.append(_denorm(z_t))

    # Denormalize
    z0 = z_t * z_std.unsqueeze(0) + z_mean.unsqueeze(0)
    z0_np = z0.detach().cpu().numpy()

    if traj_every > 0:
        # Always include final step
        traj_frames.append(z0_np)
        trajectory = np.stack(traj_frames, axis=0)  # (n_checkpoints, n_samples, latent_dim)
    else:
        trajectory = None

    return z0_np, trajectory
