from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception:  # pragma: no cover
    PyGData = None
    PyGDataLoader = None


@torch.no_grad()
def score_library(
    model,
    smiles_list,
    featurizer: Callable,
    device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_all, latents_all = [], []

    features = featurizer(smiles_list)

    is_graph = False
    if PyGData is not None and isinstance(features, list) and len(features) > 0:
        is_graph = isinstance(features[0], PyGData)

    if is_graph:
        if PyGDataLoader is None:
            raise ImportError("torch_geometric is required for graph screening.")
        loader = PyGDataLoader(features, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            pred, latent = model(batch)
            preds_all.append(pred.detach().cpu().numpy())
            latents_all.append(latent.detach().cpu().numpy())
    else:
        x = torch.tensor(np.asarray(features), dtype=torch.float32)
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
        for (xb,) in loader:
            xb = xb.to(device)
            pred, latent = model(xb)
            preds_all.append(pred.detach().cpu().numpy())
            latents_all.append(latent.detach().cpu().numpy())

    preds = np.concatenate(preds_all, axis=0) if preds_all else np.array([], dtype=float)
    latents = np.concatenate(latents_all, axis=0) if latents_all else np.empty((0, 0), dtype=float)
    return preds, latents


def rank_library(df: pd.DataFrame, score_col: str, ascending: bool = False) -> pd.DataFrame:
    if score_col not in df.columns:
        raise ValueError(f"score_col not found in DataFrame: {score_col}")
    return df.sort_values(score_col, ascending=ascending).reset_index(drop=True)


def top_k_selection(df: pd.DataFrame, k: int, score_col: str) -> pd.DataFrame:
    ranked = rank_library(df, score_col=score_col, ascending=False)
    return ranked.head(int(k)).reset_index(drop=True)
