#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_latent_umap, plot_loss_curve, plot_scatter
from src.features.descriptors import smiles_to_descriptors
from src.features.fingerprints import smiles_to_fp
from src.features.graph import MolDataset, SMEMolDataset
from src.models.gnn import AttentiveFPModel, SMERGCNModel
from src.features.graph import SME_NODE_DIM
from src.models.mlp import FingerprintMLP
from src.utils.config import parse_config_args
from src.utils.io import load_checkpoint, load_json, load_numpy, resolve_run_dir, save_json

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception:  # pragma: no cover
    PyGDataLoader = None


def resolve_device(device_cfg: str) -> torch.device:
    d = device_cfg.lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


@torch.no_grad()
def predict(model, loader, device, is_graph: bool):
    model.eval()
    preds, ys, zs = [], [], []
    for batch in loader:
        if is_graph:
            batch = batch.to(device)
            y = batch.y.view(-1)
            pred, z = model(batch)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred, z = model(x)

        preds.append(pred.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        zs.append(z.detach().cpu().numpy())

    return (
        np.concatenate(ys, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(zs, axis=0),
    )


def main():
    cfg, _ = parse_config_args("Evaluate trained model")
    data_cfg, feat_cfg, model_cfg, tr_cfg = (
        cfg["data"],
        cfg["features"],
        cfg["model"],
        cfg["training"],
    )

    device = resolve_device(str(tr_cfg.get("device", "auto")))
    run_dir = resolve_run_dir(cfg, create_if_missing=False)

    df = pd.read_csv(run_dir / "cleaned_dataset.csv")
    test_idx = load_numpy(run_dir / "splits" / "test_idx.npy")

    smiles_all = df[data_cfg["smiles_col"]].astype(str).tolist()
    y_all = df["pIC50"].astype(float).values

    feature_type = str(feat_cfg.get("type", "fingerprint")).lower()
    model_type = str(model_cfg.get("type", "mlp")).lower()
    batch_size = int(tr_cfg.get("batch_size", 64))

    if feature_type == "fingerprint":
        x_fp = smiles_to_fp(
            smiles_all,
            bits=int(feat_cfg.get("fp_bits", 2048)),
            radius=int(feat_cfg.get("fp_radius", 2)),
            use_chirality=bool(feat_cfg.get("use_chirality", False)),
        )
        if bool(feat_cfg.get("use_descriptors", False)):
            x_fp = np.concatenate([x_fp, smiles_to_descriptors(smiles_all)], axis=1)

        x = torch.tensor(x_fp[test_idx], dtype=torch.float32)
        y = torch.tensor(y_all[test_idx], dtype=torch.float32)
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

        if model_type != "mlp":
            raise ValueError("For fingerprint features, model.type must be 'mlp'.")
        model = FingerprintMLP(
            input_dim=x_fp.shape[1],
            hidden_dims=list(model_cfg.get("hidden_dims", [512, 256, 128])),
            dropout=float(model_cfg.get("dropout", 0.2)),
            activation=str(model_cfg.get("activation", "relu")),
        )
        is_graph = False

    elif feature_type == "graph":
        if PyGDataLoader is None:
            raise ImportError("torch-geometric is required for graph mode.")
        if model_type != "gnn":
            raise ValueError("For graph features, model.type must be 'gnn'.")

        ds = MolDataset([smiles_all[i] for i in test_idx], [float(y_all[i]) for i in test_idx])
        loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False)
        model = AttentiveFPModel(
            in_channels=10,
            edge_dim=6,
            hidden_dim=int(model_cfg.get("gnn_hidden", 200)),
            num_layers=int(model_cfg.get("gnn_layers", 3)),
            dropout=float(model_cfg.get("gnn_dropout", 0.1)),
        )
        is_graph = True

    elif feature_type == "sme_graph":
        if PyGDataLoader is None:
            raise ImportError("torch-geometric is required for sme_graph mode.")
        if model_type != "sme_rgcn":
            raise ValueError("For sme_graph features, model.type must be 'sme_rgcn'.")

        smask_type = str(feat_cfg.get("smask_type", "brics"))
        ds = SMEMolDataset(
            [smiles_all[i] for i in test_idx],
            [float(y_all[i]) for i in test_idx],
            smask_type=smask_type,
        )
        loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False)
        model = SMERGCNModel(
            in_feats=SME_NODE_DIM,
            hidden_feats=list(model_cfg.get("sme_hidden_feats", [200, 200])),
            ffn_hidden=int(model_cfg.get("sme_ffn_hidden", 200)),
            rgcn_dropout=float(model_cfg.get("sme_rgcn_dropout", 0.25)),
            ffn_dropout=float(model_cfg.get("sme_ffn_dropout", 0.25)),
        )
        is_graph = True

    else:
        raise ValueError(f"Unsupported features.type: {feature_type}")

    model = model.to(device)
    load_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, map_location=device)

    y_true, y_pred, latents = predict(model, loader, device=device, is_graph=is_graph)
    metrics = compute_metrics(y_true, y_pred)

    eval_dir = Path(run_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(eval_dir / "metrics.json", metrics)

    plot_scatter(y_true, y_pred, "Test: True vs Pred", eval_dir / "scatter_test.png")

    history_path = Path(run_dir) / "history.json"
    if history_path.exists():
        history = load_json(history_path)
        plot_loss_curve(history, eval_dir / "loss_curve.png")

    plot_latent_umap(latents, y_true, eval_dir / "latent_umap_test.png")

    print(f"[eval] run_dir={run_dir}")
    print(f"[eval] metrics={metrics}")


if __name__ == "__main__":
    main()
