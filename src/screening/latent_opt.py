from typing import List, Tuple

import numpy as np
import torch


def _predict_from_latent(model, z: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "out_layer"):
        return model.out_layer(z).view(-1)
    if hasattr(model, "reg_head"):
        return model.reg_head(z).view(-1)
    raise ValueError("Model does not expose a latent->prediction head (out_layer or reg_head).")


def optimize_latent_vector(model, z_init, n_steps: int, lr: float):
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
