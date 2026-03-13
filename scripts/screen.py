#!/usr/bin/env python
from pathlib import Path

import pandas as pd
import torch

from src.features.descriptors import smiles_to_descriptors
from src.features.fingerprints import smiles_to_fp
from src.features.graph import smiles_to_graph
from src.models.gnn import AttentiveFPModel
from src.models.mlp import FingerprintMLP
from src.screening.ranker import rank_library, score_library, top_k_selection
from src.screening.rerank import rerank_hook
from src.utils.config import parse_config_args
from src.utils.io import load_checkpoint, resolve_run_dir, save_numpy


def _extra_args(parser):
    parser.add_argument("--screen_csv", required=True, type=str, help="SMILES library CSV path")


def resolve_device(device_cfg: str) -> torch.device:
    d = device_cfg.lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def main():
    cfg, args = parse_config_args("Screen external library", extra_arg_fn=_extra_args)
    data_cfg, feat_cfg, model_cfg, tr_cfg, s_cfg = (
        cfg["data"],
        cfg["features"],
        cfg["model"],
        cfg["training"],
        cfg["screening"],
    )

    run_dir = resolve_run_dir(cfg, create_if_missing=False)
    device = resolve_device(str(tr_cfg.get("device", "auto")))

    lib_df = pd.read_csv(args.screen_csv)
    smiles_col = data_cfg["smiles_col"]
    if smiles_col not in lib_df.columns:
        raise ValueError(f"{smiles_col} column not found in screen CSV")
    smiles = lib_df[smiles_col].astype(str).tolist()

    feature_type = str(feat_cfg.get("type", "fingerprint")).lower()
    model_type = str(model_cfg.get("type", "mlp")).lower()

    if feature_type == "fingerprint":
        if model_type != "mlp":
            raise ValueError("For fingerprint features, model.type must be 'mlp'.")

        # Input dim inferred from training latents/preds is not sufficient; infer from feature generation.
        x_probe = smiles_to_fp(
            smiles[:1],
            bits=int(feat_cfg.get("fp_bits", 2048)),
            radius=int(feat_cfg.get("fp_radius", 2)),
            use_chirality=bool(feat_cfg.get("use_chirality", False)),
        )
        if bool(feat_cfg.get("use_descriptors", False)):
            x_probe = pd.concat(
                [
                    pd.DataFrame(x_probe),
                    pd.DataFrame(smiles_to_descriptors(smiles[:1])),
                ],
                axis=1,
            ).values

        model = FingerprintMLP(
            input_dim=x_probe.shape[1],
            hidden_dims=list(model_cfg.get("hidden_dims", [512, 256, 128])),
            dropout=float(model_cfg.get("dropout", 0.2)),
            activation=str(model_cfg.get("activation", "relu")),
        )

        def featurizer(smis):
            x = smiles_to_fp(
                smis,
                bits=int(feat_cfg.get("fp_bits", 2048)),
                radius=int(feat_cfg.get("fp_radius", 2)),
                use_chirality=bool(feat_cfg.get("use_chirality", False)),
            )
            if bool(feat_cfg.get("use_descriptors", False)):
                x = pd.concat([pd.DataFrame(x), pd.DataFrame(smiles_to_descriptors(smis))], axis=1).values
            return x

    elif feature_type == "graph":
        if model_type != "gnn":
            raise ValueError("For graph features, model.type must be 'gnn'.")
        model = AttentiveFPModel(
            in_channels=10,
            edge_dim=6,
            hidden_dim=int(model_cfg.get("gnn_hidden", 200)),
            num_layers=int(model_cfg.get("gnn_layers", 3)),
            dropout=float(model_cfg.get("gnn_dropout", 0.1)),
        )

        def featurizer(smis):
            return [smiles_to_graph(s) for s in smis]

    else:
        raise ValueError(f"Unsupported features.type: {feature_type}")

    model = model.to(device)
    load_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, map_location=device)

    preds, latents = score_library(
        model=model,
        smiles_list=smiles,
        featurizer=featurizer,
        device=device,
        batch_size=int(tr_cfg.get("batch_size", 64)),
    )

    score_col = str(s_cfg.get("score_col", "pred_pIC50"))
    reranked = rerank_hook(preds, metadata={"n": len(preds)})

    out_df = lib_df.copy()
    out_df[score_col] = reranked

    ranked = rank_library(out_df, score_col=score_col, ascending=False)
    top_k = top_k_selection(ranked, int(s_cfg.get("top_k", 100)), score_col=score_col)

    out_dir = Path(run_dir) / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked.to_csv(out_dir / "ranked_all.csv", index=False)
    top_k.to_csv(out_dir / "top_k.csv", index=False)
    save_numpy(out_dir / "screen_latents.npy", latents)

    print(f"[screen] run_dir={run_dir}")
    print(f"[screen] saved: {out_dir / 'ranked_all.csv'}")
    print(f"[screen] saved: {out_dir / 'top_k.csv'}")


if __name__ == "__main__":
    main()
