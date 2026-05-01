#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_latent_tsne, plot_latent_umap, plot_loss_curve, plot_scatter
from src.features.descriptors import smiles_to_descriptors
from src.features.fingerprints import smiles_to_fp
from src.features.graph import MolDataset, SMEMolDataset, SME_NODE_DIM
from src.features.unimol import smiles_to_unimol
from src.models.gnn import AttentiveFPModel, SMERGCNModel
from src.models.mlp import FingerprintMLP
from src.models.vqc_module import VQCEncoderHead
from src.utils.config import parse_config_args
from src.utils.io import load_checkpoint, load_json, load_numpy, resolve_run_dir, save_json


class _VQCModel(nn.Module):
    def __init__(self, backbone: nn.Module, vqc_head: VQCEncoderHead):
        super().__init__()
        self.backbone = backbone
        self.vqc_head = vqc_head
        self.out_layer = nn.Linear(vqc_head.n_qubits, 1)

    def get_latent_dim(self) -> int:
        return self.vqc_head.n_qubits

    def forward(self, x):
        _, z = self.backbone(x)
        z_vqc = self.vqc_head(z)
        pred = self.out_layer(z_vqc)
        return pred.view(-1), z_vqc

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
    splits_dir = run_dir / "splits"
    train_idx = load_numpy(splits_dir / "train_idx.npy")
    val_idx   = load_numpy(splits_dir / "val_idx.npy")
    _test_path = splits_dir / "test_idx.npy"
    test_idx  = load_numpy(_test_path) if _test_path.exists() else None

    smiles_all = df[data_cfg["smiles_col"]].astype(str).tolist()
    label_col = data_cfg.get("label_col", "pIC50")
    y_all = df[label_col].astype(float).values

    # If encoder was trained with normalize_y, apply same normalization so that
    # metrics and plots are computed in the same scale as model predictions.
    _scaler_path = run_dir / "y_scaler.json"
    normalize_y = bool(tr_cfg.get("normalize_y", False))
    if normalize_y and _scaler_path.exists():
        _scaler = load_json(_scaler_path)
        y_mean, y_std = float(_scaler["mean"]), float(_scaler["std"])
        y_all = (y_all - y_mean) / y_std
        print(f"[evaluate] normalize_y applied: mean={y_mean:.4f} std={y_std:.4f}")
    else:
        y_mean, y_std = 0.0, 1.0

    feature_type = str(feat_cfg.get("type", "fingerprint")).lower()
    model_type   = str(model_cfg.get("type", "mlp")).lower()
    batch_size   = int(tr_cfg.get("batch_size", 64))
    latent_dim   = int(model_cfg.get("latent_dim", 128))
    use_vqc      = bool(model_cfg.get("use_vqc", False))

    if feature_type in ("fingerprint", "unimol"):
        if model_type != "mlp":
            raise ValueError(f"For {feature_type} features, model.type must be 'mlp'.")
        if feature_type == "fingerprint":
            x_fp = smiles_to_fp(
                smiles_all,
                bits=int(feat_cfg.get("fp_bits", 2048)),
                radius=int(feat_cfg.get("fp_radius", 2)),
                use_chirality=bool(feat_cfg.get("use_chirality", False)),
            )
            if bool(feat_cfg.get("use_descriptors", False)):
                x_fp = np.concatenate([x_fp, smiles_to_descriptors(smiles_all)], axis=1)
        else:
            print("[evaluate] extracting Uni-Mol embeddings (one-time)...")
            x_fp = smiles_to_unimol(smiles_all)

        def make_loader(idx):
            x = torch.tensor(x_fp[idx], dtype=torch.float32)
            y = torch.tensor(y_all[idx], dtype=torch.float32)
            return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

        hidden_dims = list(model_cfg.get("hidden_dims", [512, 256, 128]))
        if not use_vqc:
            hidden_dims[-1] = latent_dim
        backbone = FingerprintMLP(
            input_dim=x_fp.shape[1],
            hidden_dims=hidden_dims,
            dropout=float(model_cfg.get("dropout", 0.2)),
            activation=str(model_cfg.get("activation", "relu")),
        )
        is_graph = False

    elif feature_type == "graph":
        if PyGDataLoader is None:
            raise ImportError("torch-geometric is required for graph mode.")
        if model_type != "gnn":
            raise ValueError("For graph features, model.type must be 'gnn'.")

        def make_loader(idx):
            ds = MolDataset([smiles_all[i] for i in idx], [float(y_all[i]) for i in idx])
            return PyGDataLoader(ds, batch_size=batch_size, shuffle=False)

        gnn_hidden = int(model_cfg.get("gnn_hidden", 256)) if use_vqc else latent_dim
        backbone = AttentiveFPModel(
            in_channels=10,
            edge_dim=6,
            hidden_dim=gnn_hidden,
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

        def make_loader(idx):
            ds = SMEMolDataset(
                [smiles_all[i] for i in idx],
                [float(y_all[i]) for i in idx],
                smask_type=smask_type,
            )
            return PyGDataLoader(ds, batch_size=batch_size, shuffle=False)

        _ffn_dims_cfg = model_cfg.get("sme_ffn_dims")
        if _ffn_dims_cfg is not None:
            sme_kwargs = {"ffn_dims": [int(d) for d in _ffn_dims_cfg] + [latent_dim]}
        else:
            sme_ffn = int(model_cfg.get("sme_ffn_hidden", 256)) if use_vqc else latent_dim
            sme_kwargs = {"ffn_hidden": sme_ffn}
        backbone = SMERGCNModel(
            in_feats=SME_NODE_DIM,
            hidden_feats=list(model_cfg.get("sme_hidden_feats", [256, 256])),
            rgcn_dropout=float(model_cfg.get("sme_rgcn_dropout", 0.25)),
            ffn_dropout=float(model_cfg.get("sme_ffn_dropout", 0.25)),
            **sme_kwargs,
        )
        is_graph = True

    else:
        raise ValueError(f"Unsupported features.type: {feature_type}")

    if use_vqc:
        vqc_head = VQCEncoderHead(latent_dim=backbone.get_latent_dim(), n_qubits=latent_dim)
        model = _VQCModel(backbone, vqc_head)
    else:
        model = backbone

    model = model.to(device)
    load_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, map_location=device)

    eval_dir = Path(run_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train_idx, "val": val_idx}
    if test_idx is not None:
        splits["test"] = test_idx
    all_latents, all_y_true, all_split_labels = [], [], []

    for split_name, idx in splits.items():
        loader = make_loader(idx)
        y_true, y_pred, latents = predict(model, loader, device=device, is_graph=is_graph)
        metrics = compute_metrics(y_true, y_pred)

        save_json(eval_dir / f"metrics_{split_name}.json", metrics)
        plot_scatter(
            y_true, y_pred,
            f"{split_name.capitalize()} — True vs Pred (normalized)",
            eval_dir / f"scatter_{split_name}.png",
            label=label_col,
        )
        if normalize_y:
            y_true_orig = y_true * y_std + y_mean
            y_pred_orig = y_pred * y_std + y_mean
            plot_scatter(
                y_true_orig, y_pred_orig,
                f"{split_name.capitalize()} — True vs Pred",
                eval_dir / f"scatter_{split_name}_orig.png",
                label=label_col,
            )
        plot_latent_tsne(latents, y_true, eval_dir / f"tsne_{split_name}.png", split_name=split_name)
        plot_latent_umap(latents, y_true, eval_dir / f"umap_{split_name}.png", split_name=split_name)

        # Save latents, SMILES, and y for downstream use (diffusion sampling / screening)
        np.save(run_dir / f"latents_{split_name}.npy", latents)
        smiles_split = [smiles_all[i] for i in idx]
        np.save(run_dir / f"smiles_{split_name}.npy", np.array(smiles_split, dtype=object))
        np.save(run_dir / f"y_{split_name}.npy", y_true)

        all_latents.append(latents)
        all_y_true.append(y_true)
        all_split_labels.extend([split_name] * len(y_true))

        print(f"[eval] {split_name:5s}  metrics={metrics}")

    # Combined t-SNE coloured by split
    combined_latents = np.concatenate(all_latents, axis=0)
    combined_y = np.concatenate(all_y_true, axis=0)
    plot_latent_tsne(combined_latents, combined_y, eval_dir / "tsne_all.png", split_name="all splits")
    plot_latent_umap(combined_latents, combined_y, eval_dir / "umap_all.png", split_name="all splits")

    history_path = Path(run_dir) / "history.json"
    if history_path.exists():
        history = load_json(history_path)
        plot_loss_curve(history, eval_dir / "loss_curve.png")

    print(f"[eval] run_dir={run_dir}")


if __name__ == "__main__":
    main()
