#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.features.descriptors import smiles_to_descriptors
from src.features.fingerprints import smiles_to_fp
from src.features.graph import MolDataset, SMEMolDataset, SME_NODE_DIM
from src.models.gnn import AttentiveFPModel, SMERGCNModel
from src.models.mlp import FingerprintMLP
from src.training.scheduler import get_scheduler
from src.training.trainer import Trainer
from src.utils.config import parse_config_args
from src.utils.io import load_numpy, resolve_run_dir, save_json, save_numpy, save_predictions_csv
from src.utils.seed import set_seed

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
    preds, latents, ys = [], [], []
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
        latents.append(z.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())

    return (
        np.concatenate(ys, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(latents, axis=0),
    )


def build_fp_features(smiles, feat_cfg):
    x_fp = smiles_to_fp(
        smiles,
        bits=int(feat_cfg.get("fp_bits", 2048)),
        radius=int(feat_cfg.get("fp_radius", 2)),
        use_chirality=bool(feat_cfg.get("use_chirality", False)),
    )
    if bool(feat_cfg.get("use_descriptors", False)):
        x_desc = smiles_to_descriptors(smiles)
        x_fp = np.concatenate([x_fp, x_desc], axis=1)
    return x_fp


def main():
    cfg, _ = parse_config_args("Train ligand-only ML model")
    data_cfg, feat_cfg, model_cfg, tr_cfg = (
        cfg["data"],
        cfg["features"],
        cfg["model"],
        cfg["training"],
    )

    set_seed(int(tr_cfg.get("seed", 42)))
    device = resolve_device(str(tr_cfg.get("device", "auto")))

    run_dir = resolve_run_dir(cfg, create_if_missing=False)
    df = pd.read_csv(run_dir / "cleaned_dataset.csv")
    train_idx = load_numpy(run_dir / "splits" / "train_idx.npy")
    val_idx = load_numpy(run_dir / "splits" / "val_idx.npy")
    test_idx = load_numpy(run_dir / "splits" / "test_idx.npy")

    smiles_col = data_cfg["smiles_col"]
    smiles_all = df[smiles_col].astype(str).tolist()
    y_all = df["pIC50"].astype(float).values

    feature_type = str(feat_cfg.get("type", "fingerprint")).lower()
    model_type = str(model_cfg.get("type", "mlp")).lower()
    batch_size = int(tr_cfg.get("batch_size", 64))
    latent_dim = int(model_cfg.get("latent_dim", 128))

    if feature_type == "fingerprint":
        x_all = build_fp_features(smiles_all, feat_cfg)

        def make_loader(indices, shuffle=False):
            x = torch.tensor(x_all[indices], dtype=torch.float32)
            y = torch.tensor(y_all[indices], dtype=torch.float32)
            return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

        train_loader = make_loader(train_idx, shuffle=True)
        val_loader = make_loader(val_idx, shuffle=False)

        pred_train_loader = make_loader(train_idx, shuffle=False)
        pred_val_loader = make_loader(val_idx, shuffle=False)
        pred_test_loader = make_loader(test_idx, shuffle=False)

        if model_type != "mlp":
            raise ValueError("For fingerprint features, model.type must be 'mlp'.")

        hidden_dims = list(model_cfg.get("hidden_dims", [512, 256, 128]))
        hidden_dims[-1] = latent_dim
        model = FingerprintMLP(
            input_dim=x_all.shape[1],
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

        def make_g_loader(indices, shuffle=False):
            smiles = [smiles_all[i] for i in indices]
            y = [float(y_all[i]) for i in indices]
            ds = MolDataset(smiles, y)
            return PyGDataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        train_loader = make_g_loader(train_idx, shuffle=True)
        val_loader = make_g_loader(val_idx, shuffle=False)

        pred_train_loader = make_g_loader(train_idx, shuffle=False)
        pred_val_loader = make_g_loader(val_idx, shuffle=False)
        pred_test_loader = make_g_loader(test_idx, shuffle=False)

        model = AttentiveFPModel(
            in_channels=10,   # DEFAULT_NODE_DIM
            edge_dim=6,       # DEFAULT_EDGE_DIM
            hidden_dim=latent_dim,
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
        use_chirality = bool(feat_cfg.get("use_chirality", True))

        def make_sme_loader(indices, shuffle=False):
            smiles = [smiles_all[i] for i in indices]
            y = [float(y_all[i]) for i in indices]
            ds = SMEMolDataset(smiles, y, smask_type=smask_type, use_chirality=use_chirality)
            return PyGDataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        train_loader = make_sme_loader(train_idx, shuffle=True)
        val_loader = make_sme_loader(val_idx, shuffle=False)

        pred_train_loader = make_sme_loader(train_idx, shuffle=False)
        pred_val_loader = make_sme_loader(val_idx, shuffle=False)
        pred_test_loader = make_sme_loader(test_idx, shuffle=False)

        model = SMERGCNModel(
            in_feats=SME_NODE_DIM,
            hidden_feats=list(model_cfg.get("sme_hidden_feats", [200, 200])),
            ffn_hidden=latent_dim,
            rgcn_dropout=float(model_cfg.get("sme_rgcn_dropout", 0.25)),
            ffn_dropout=float(model_cfg.get("sme_ffn_dropout", 0.25)),
        )
        is_graph = True

    else:
        raise ValueError(f"Unsupported features.type: {feature_type}")

    model = model.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(tr_cfg.get("lr", 1e-3)),
        weight_decay=float(tr_cfg.get("weight_decay", 1e-5)),
    )
    criterion = nn.MSELoss()
    scheduler = get_scheduler(optimizer, cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        is_graph=is_graph,
        early_stopping_patience=int(tr_cfg.get("early_stopping_patience", 15)),
        checkpoint_every=int(tr_cfg.get("checkpoint_every", 10)),
        run_dir=run_dir,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(tr_cfg.get("epochs", 100)),
    )
    save_json(run_dir / "history.json", history)

    outputs_dir = Path(run_dir) / "predictions"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": (train_idx, pred_train_loader),
        "val": (val_idx, pred_val_loader),
        "test": (test_idx, pred_test_loader),
    }

    for split, (idx_arr, loader) in split_map.items():
        y_true, y_pred, z = predict(model, loader, device=device, is_graph=is_graph)
        split_smiles = [smiles_all[i] for i in idx_arr]

        pred_df = pd.DataFrame(
            {
                "smiles": split_smiles,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        save_predictions_csv(outputs_dir / f"{split}_preds.csv", pred_df)
        save_numpy(run_dir / f"latents_{split}.npy", z)

    print(f"[train] done. run_dir={run_dir}")
    print(f"[train] best_epoch={history.get('best_epoch')} best_val={history.get('best_val_loss'):.4f}")


if __name__ == "__main__":
    main()
