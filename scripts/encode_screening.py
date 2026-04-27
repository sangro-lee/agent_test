#!/usr/bin/env python
"""
Encode an external screening library into latent vectors for use with sample_cfg.py.

Given a CSV of SMILES (no pIC50 required), loads a trained encoder and extracts
latent representations.  Saves two .npy files:
  - <out_dir>/latents_screening.npy   float32 (N, latent_dim)
  - <out_dir>/smiles_screening.npy    object  (N,)

Usage:
  python scripts/encode_screening.py \\
      --config outputs/runs/sme_random/config_resolved.yaml \\
      --screening_csv data/screening_library.csv \\
      --smiles_col smiles \\
      --out_dir outputs/runs/sme_random/screening

Then pass to sample_cfg.py:
  python scripts/sample_cfg.py \\
      --config outputs/runs/sme_random/config_resolved.yaml \\
      --screening_latents outputs/runs/sme_random/screening/latents_screening.npy \\
      --screening_smiles  outputs/runs/sme_random/screening/smiles_screening.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.features.descriptors import smiles_to_descriptors
from src.features.fingerprints import smiles_to_fp
from src.features.graph import MolDataset, SMEMolDataset, SME_NODE_DIM
from src.models.gnn import AttentiveFPModel, SMERGCNModel
from src.models.mlp import FingerprintMLP
from src.utils.config import parse_config_args
from src.utils.io import load_checkpoint, resolve_run_dir

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception:
    PyGDataLoader = None


def _extra_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--screening_csv", type=str, required=True,
                        help="Path to CSV file with external SMILES to encode")
    parser.add_argument("--smiles_col", type=str, default=None,
                        help="Column name for SMILES (default: same as config data.smiles_col)")
    parser.add_argument("--pic50_col", type=str, default="pIC50",
                        help="Column name for pIC50 values (default: pIC50). "
                             "If found, saves y_screening.npy for use in sample_cfg.py")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: <run_dir>/screening)")
    parser.add_argument("--batch_size", type=int, default=64)


@torch.no_grad()
def _encode(model, loader, device: torch.device, is_graph: bool) -> np.ndarray:
    model.eval()
    zs = []
    for batch in loader:
        if is_graph:
            batch = batch.to(device)
            _, z = model(batch)
        else:
            x, = batch  # no label
            x = x.to(device)
            _, z = model(x)
        zs.append(z.detach().cpu().numpy())
    return np.concatenate(zs, axis=0)


def main() -> None:
    cfg, args = parse_config_args(
        "Encode external screening library to latents",
        extra_arg_fn=_extra_args,
    )

    data_cfg  = cfg["data"]
    feat_cfg  = cfg["features"]
    model_cfg = cfg["model"]
    tr_cfg    = cfg["training"]

    # Device
    device_str = str(tr_cfg.get("device", "auto")).lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    run_dir = resolve_run_dir(cfg, create_if_missing=False)
    out_dir = Path(args.out_dir) if args.out_dir else Path(run_dir) / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load screening CSV
    scr_df = pd.read_csv(args.screening_csv)
    smiles_col = args.smiles_col or str(data_cfg["smiles_col"])
    if smiles_col not in scr_df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {args.screening_csv}. "
                         f"Available: {list(scr_df.columns)}")
    smiles_list = scr_df[smiles_col].astype(str).tolist()
    print(f"[encode_screening] {len(smiles_list)} molecules from {args.screening_csv}")

    feature_type = str(feat_cfg.get("type", "fingerprint")).lower()
    batch_size   = int(args.batch_size)

    # Build loader and model
    if feature_type == "fingerprint":
        x_fp = smiles_to_fp(
            smiles_list,
            bits=int(feat_cfg.get("fp_bits", 2048)),
            radius=int(feat_cfg.get("fp_radius", 2)),
            use_chirality=bool(feat_cfg.get("use_chirality", False)),
        )
        if bool(feat_cfg.get("use_descriptors", False)):
            x_fp = np.concatenate([x_fp, smiles_to_descriptors(smiles_list)], axis=1)

        loader = DataLoader(
            TensorDataset(torch.tensor(x_fp, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False,
        )
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
        # Dummy labels (not used)
        ds = MolDataset(smiles_list, [0.0] * len(smiles_list))
        loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False)
        model = AttentiveFPModel(
            in_channels=10, edge_dim=6,
            hidden_dim=int(model_cfg.get("gnn_hidden", 200)),
            num_layers=int(model_cfg.get("gnn_layers", 3)),
            dropout=float(model_cfg.get("gnn_dropout", 0.1)),
        )
        is_graph = True

    elif feature_type == "sme_graph":
        if PyGDataLoader is None:
            raise ImportError("torch-geometric is required for sme_graph mode.")
        smask_type = str(feat_cfg.get("smask_type", "brics"))
        ds = SMEMolDataset(smiles_list, [0.0] * len(smiles_list), smask_type=smask_type)
        loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False)
        _ffn_dims_cfg = model_cfg.get("sme_ffn_dims")
        if _ffn_dims_cfg is not None:
            _latent_dim = int(model_cfg.get("latent_dim", 8))
            sme_kwargs = {"ffn_dims": [int(d) for d in _ffn_dims_cfg] + [_latent_dim]}
        else:
            sme_kwargs = {"ffn_hidden": int(model_cfg.get("sme_ffn_hidden", 200))}
        model = SMERGCNModel(
            in_feats=SME_NODE_DIM,
            hidden_feats=list(model_cfg.get("sme_hidden_feats", [200, 200])),
            rgcn_dropout=float(model_cfg.get("sme_rgcn_dropout", 0.25)),
            ffn_dropout=float(model_cfg.get("sme_ffn_dropout", 0.25)),
            **sme_kwargs,
        )
        is_graph = True

    else:
        raise ValueError(f"Unsupported features.type: {feature_type}")

    model = model.to(device)
    load_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, map_location=device_str)

    # Encode
    latents = _encode(model, loader, device, is_graph)

    # Save
    np.save(out_dir / "latents_screening.npy", latents)
    np.save(out_dir / "smiles_screening.npy", np.array(smiles_list, dtype=object))

    print(f"[encode_screening] latents shape: {latents.shape}")
    print(f"[encode_screening] saved → {out_dir}/latents_screening.npy")
    print(f"[encode_screening] saved → {out_dir}/smiles_screening.npy")

    if args.pic50_col in scr_df.columns:
        y_scr = scr_df[args.pic50_col].values.astype(np.float32)
        np.save(out_dir / "y_screening.npy", y_scr)
        print(f"[encode_screening] saved → {out_dir}/y_screening.npy")
    else:
        print(f"[encode_screening] '{args.pic50_col}' column not found — y_screening.npy not saved")

    print(f"\nTo use in sample_cfg.py, add:")
    print(f"  --screening_latents {out_dir}/latents_screening.npy \\")
    print(f"  --screening_smiles  {out_dir}/smiles_screening.npy \\")
    print(f"  --screening_pic50   {out_dir}/y_screening.npy")


if __name__ == "__main__":
    main()
