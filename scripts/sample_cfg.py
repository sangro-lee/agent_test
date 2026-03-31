#!/usr/bin/env python
"""
Sample molecules using a trained CFG diffusion denoiser.

Loads:
  outputs/runs/<exp>/diffusion/denoiser_cfg.pt
  outputs/runs/<exp>/latents_train.npy

Saves to:
  outputs/runs/<exp>/diffusion/cfg_w{guidance_scale}/top_candidates.csv
  outputs/runs/<exp>/diffusion/cfg_w{guidance_scale}/z_samples.npy

Usage:
  python scripts/sample_cfg.py --config /tmp/sme_random_resolved.yaml \\
      --guidance_scale 3.0 --target_pic50 8.0

  # Try multiple guidance scales
  for w in 1.0 2.0 3.0 5.0; do
    python scripts/sample_cfg.py --config /tmp/sme_random_resolved.yaml \\
        --guidance_scale $w --target_pic50 8.0
  done
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.diffusion import ConditionalDenoisingMLP
from src.models.vqc_diffusion import AngleVQCDenoiser, HybridUNetDenoiser, QubitCondVQCDenoiser
from src.models.vqc_module import VQCConditionalDenoiser
from src.screening.latent_opt import retrieve_nearest, sample_cfg
from src.utils.config import parse_config_args
from src.utils.io import load_numpy, resolve_run_dir
from src.utils.seed import set_seed


def _extra_args(parser):
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="CFG weight w (0=uncond, 1=cond, >1=amplified)")
    parser.add_argument("--target_pic50",   type=float, default=None,
                        help="Target pIC50 in original scale (default: 90th pct of train)")
    parser.add_argument("--n_samples",      type=int,   default=500)
    parser.add_argument("--top_k",          type=int,   default=50)
    parser.add_argument("--sampler",        type=str,   default="ddim",
                        choices=["ddim", "ddpm"])
    parser.add_argument("--T",              type=int,   default=1000,
                        help="Diffusion steps used during training (must match checkpoint)")
    parser.add_argument("--diff_epochs",    type=int,   default=200,
                        help="Training epochs used (must match checkpoint folder)")
    parser.add_argument("--date",           type=str,   default=None,
                        help="Date subfolder under diffusion/ (e.g. 2026-03-26); omit for legacy paths")
    parser.add_argument("--screening_latents", type=str, default=None,
                        help="Path to .npy file of external screening latents")
    parser.add_argument("--screening_smiles",  type=str, default=None,
                        help="Path to .npy file of external screening SMILES (object array)")


def main():
    cfg, args = parse_config_args(
        "Sample with CFG diffusion denoiser",
        extra_arg_fn=_extra_args,
    )
    tr_cfg = cfg["training"]
    set_seed(int(tr_cfg.get("seed", 42)))

    run_dir = resolve_run_dir(cfg, create_if_missing=False)
    device = str(tr_cfg.get("device", "auto"))
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # ---- Load saved denoiser ----------------------------------------------
    _diff_subdir = (Path("diffusion") / args.date / f"T{args.T}_ep{args.diff_epochs}"
                    if args.date else Path("diffusion") / f"T{args.T}_ep{args.diff_epochs}")
    ckpt_path = Path(run_dir) / _diff_subdir / "denoiser_cfg.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Denoiser not found: {ckpt_path}\n"
            f"Run train_diffusion.py first."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    T = int(ckpt["T"])
    model_type = ckpt.get("model_type", "mlp")
    denoiser_type = ckpt.get("denoiser_type", model_type)  # fallback for old checkpoints

    if model_type == "vqc":
        denoiser = VQCConditionalDenoiser(
            latent_dim=latent_dim,
            n_qubits=int(ckpt.get("n_qubits", latent_dim)),
            n_layers=int(ckpt.get("n_layers", 2)),
        )
    elif denoiser_type in ("vqc_angle", "vqc_angle_delta", "vqc_angle_reupload"):
        denoiser = AngleVQCDenoiser(
            latent_dim=latent_dim,
            n_layers=int(ckpt.get("n_layers", 2)),
            num_blocks=int(ckpt.get("num_blocks", 6)),
            time_dim=int(ckpt.get("time_dim", 32)),
            cond_dim=int(ckpt.get("cond_dim", 32)),
            use_delta=bool(ckpt.get("use_delta", False)),
            use_reupload=bool(ckpt.get("use_reupload", False)),
            initial_cnot=bool(ckpt.get("initial_cnot", False)),
            full_encoding=bool(ckpt.get("full_encoding", False)),
        )
    elif denoiser_type == "vqc_qubit_cond":
        denoiser = QubitCondVQCDenoiser(
            latent_dim=latent_dim,
            n_layers=int(ckpt.get("n_layers", 2)),
            num_blocks=int(ckpt.get("num_blocks", 6)),
        )
    elif denoiser_type in ("unet", "unet_vqc"):
        denoiser = HybridUNetDenoiser(
            latent_dim=latent_dim,
            unet_dims=ckpt.get("unet_dims"),
            n_layers=int(ckpt.get("n_layers", 2)),
            time_dim=int(ckpt.get("time_dim", 128)),
            cond_dim=int(ckpt.get("cond_dim", 128)),
            use_vqc=(denoiser_type == "unet_vqc"),
        )
    elif denoiser_type == "mlp_ortho":
        denoiser = ConditionalDenoisingMLP(
            latent_dim=latent_dim,
            time_dim=int(ckpt["time_dim"]),
            cond_dim=int(ckpt["cond_dim"]),
            hidden_dim=int(ckpt.get("hidden_dim", 512)),
            use_orthogonal=True,
        )
    else:
        denoiser = ConditionalDenoisingMLP(
            latent_dim=latent_dim,
            time_dim=int(ckpt["time_dim"]),
            cond_dim=int(ckpt["cond_dim"]),
            hidden_dim=int(ckpt.get("hidden_dim", 512)),
        )
    denoiser.load_state_dict(ckpt["state_dict"])
    denoiser.set_normalization(
        z_mean=torch.tensor(ckpt["z_mean"]),
        z_std=torch.tensor(ckpt["z_std"]),
        c_mean=torch.tensor([ckpt["c_mean"]]),
        c_std=torch.tensor([ckpt["c_std"]]),
    )

    # ---- Load latents for nearest-neighbor retrieval (train / val / test) ----
    df = pd.read_csv(run_dir / "cleaned_dataset.csv")
    splits_dir = run_dir / "splits"
    train_idx = load_numpy(splits_dir / "train_idx.npy")
    y_all = df[cfg["data"]["activity_col"]].astype(float).values
    y_train = y_all[train_idx]

    # Build per-split pools: {split_name: (z_array, smiles_list, y_lookup)}
    # y_lookup: {smiles: actual_pIC50} — empty dict for pools without known activity
    retrieval_pools: dict = {}
    for split_name in ("train", "val", "test"):
        z_path = run_dir / f"latents_{split_name}.npy"
        s_path = run_dir / f"smiles_{split_name}.npy"
        y_path = run_dir / f"y_{split_name}.npy"
        if z_path.exists() and s_path.exists():
            smiles_pool = list(np.load(s_path, allow_pickle=True).astype(str))
            y_lookup = {}
            if y_path.exists():
                y_pool = np.load(y_path).astype(np.float32)
                y_lookup = dict(zip(smiles_pool, y_pool.tolist()))
            retrieval_pools[split_name] = (
                load_numpy(z_path).astype(np.float32),
                smiles_pool,
                y_lookup,
            )
        else:
            print(f"[sample_cfg] Warning: {split_name} latents not found, skipping. "
                  f"Re-run evaluate.py to generate them.")

    # Fallback: if none found, load legacy latents_train.npy
    if not retrieval_pools:
        smiles_all = df[cfg["data"]["smiles_col"]].astype(str).tolist()
        smiles_train = [smiles_all[i] for i in train_idx]
        z_fallback = load_numpy(run_dir / "latents_train.npy").astype(np.float32)
        retrieval_pools["train"] = (z_fallback, smiles_train, {})

    # Optional: external screening pool (no actual pIC50 available)
    if args.screening_latents and args.screening_smiles:
        z_scr = np.load(args.screening_latents).astype(np.float32)
        s_scr = list(np.load(args.screening_smiles, allow_pickle=True).astype(str))
        retrieval_pools["screening"] = (z_scr, s_scr, {})
        print(f"[sample_cfg] Loaded screening pool: {len(s_scr)} molecules")

    # ---- Determine target pIC50 -------------------------------------------
    target_pic50 = args.target_pic50
    if target_pic50 is None:
        target_pic50 = float(np.percentile(y_train, 90))
    print(f"[sample_cfg] target_pIC50={target_pic50:.3f}  guidance_scale={args.guidance_scale}")
    print(f"[sample_cfg] n_samples={args.n_samples}  sampler={args.sampler}  T={T}")

    # ---- Sample -----------------------------------------------------------
    z_samples = sample_cfg(
        denoiser=denoiser,
        target_pic50=target_pic50,
        latent_dim=latent_dim,
        n_samples=args.n_samples,
        T=T,
        guidance_scale=args.guidance_scale,
        sampler=args.sampler,
        device=device,
    )

    # Latent diversity diagnostic (compare to train latents)
    z_std_per_dim = z_samples.std(axis=0)
    _train_latent_path = Path(run_dir) / "latents_train.npy"
    if _train_latent_path.exists():
        _z_train = np.load(_train_latent_path)
        _train_std = float(_z_train.std(axis=0).mean())
        _ratio = float(z_std_per_dim.mean()) / (_train_std + 1e-8)
        _diversity_data = {"mean_std": float(z_std_per_dim.mean()),
                           "train_std": _train_std, "ratio": _ratio}
        print(f"[sample_cfg] latent diversity: mean_std={_diversity_data['mean_std']:.4f}  "
              f"train_std={_train_std:.4f}  ratio={_ratio:.3f}  "
              f"({'OK' if _ratio > 0.7 else 'low' if _ratio > 0.3 else 'COLLAPSE'})")
    else:
        _diversity_data = {"mean_std": float(z_std_per_dim.mean()),
                           "train_std": None, "ratio": None}
        print(f"[sample_cfg] latent diversity: mean_std={_diversity_data['mean_std']:.4f}  "
              f"min_std={float(z_std_per_dim.min()):.4f}  max_std={float(z_std_per_dim.max()):.4f}")

    # ---- Score via reg_head (last linear layer of encoder) ----------------
    # Load encoder model to score sampled latents
    from src.utils.io import load_checkpoint

    # One probe SMILES for building model input dimension
    _first_pool = next(iter(retrieval_pools.values()))
    _probe_smiles = _first_pool[1][:1]  # smiles_pool[0:1]

    # We need the original encoder's reg_head — load it
    try:
        from src.utils.io import load_checkpoint
        from src.features.descriptors import smiles_to_descriptors
        from src.features.fingerprints import smiles_to_fp
        from src.features.graph import SME_NODE_DIM
        from src.models.gnn import AttentiveFPModel, SMERGCNModel
        from src.models.mlp import FingerprintMLP

        feat_cfg  = cfg["features"]
        model_cfg = cfg["model"]
        feature_type = str(feat_cfg.get("type", "fingerprint")).lower()

        if feature_type == "fingerprint":
            x_probe = smiles_to_fp(_probe_smiles,
                bits=int(feat_cfg.get("fp_bits", 2048)),
                radius=int(feat_cfg.get("fp_radius", 2)),
                use_chirality=bool(feat_cfg.get("use_chirality", False)))
            if bool(feat_cfg.get("use_descriptors", False)):
                x_probe = np.concatenate([x_probe, smiles_to_descriptors(_probe_smiles)], axis=1)
            model = FingerprintMLP(input_dim=x_probe.shape[1],
                hidden_dims=list(model_cfg.get("hidden_dims", [512, 256, 128])),
                dropout=float(model_cfg.get("dropout", 0.2)))
        elif feature_type == "graph":
            model = AttentiveFPModel(in_channels=10, edge_dim=6,
                hidden_dim=int(model_cfg.get("gnn_hidden", 200)),
                num_layers=int(model_cfg.get("gnn_layers", 3)))
        elif feature_type == "sme_graph":
            _ffn_dims_cfg = model_cfg.get("sme_ffn_dims")
            if _ffn_dims_cfg is not None:
                sme_kwargs = {"ffn_dims": [int(d) for d in _ffn_dims_cfg] + [latent_dim]}
            else:
                use_vqc_enc = str(model_cfg.get("type", "")).lower() == "vqc"
                sme_kwargs = {"ffn_hidden": int(model_cfg.get("sme_ffn_hidden", 256)) if use_vqc_enc else latent_dim}
            model = SMERGCNModel(in_feats=SME_NODE_DIM,
                hidden_feats=list(model_cfg.get("sme_hidden_feats", [200, 200])),
                **sme_kwargs)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        load_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, map_location=device)
        model.eval()

        reg_head = model.out_layer if hasattr(model, "out_layer") else model.reg_head
        device_t = torch.device(device)
        reg_head = reg_head.to(device_t)
        z_tensor = torch.tensor(z_samples, dtype=torch.float32, device=device_t)
        with torch.no_grad():
            pred_batch = reg_head(z_tensor).view(-1).cpu().numpy()
    except Exception as e:
        print(f"[sample_cfg] Warning: could not load encoder for scoring ({e})")
        print("[sample_cfg] Scores set to 0.")
        pred_batch = np.zeros(len(z_samples))

    # ---- Retrieve nearest molecules per split / screening pool ------------
    order = np.argsort(-pred_batch)
    top_indices = order[: min(args.top_k, len(order))]
    print(f"[sample_cfg] top-{len(top_indices)} / {len(z_samples)} latents  "
          f"pred range: {pred_batch[top_indices].min():.3f} ~ {pred_batch[top_indices].max():.3f}")

    # Diversity of top-k selected latents (the actual candidates)
    _z_topk = z_samples[top_indices]
    _topk_std = float(_z_topk.std(axis=0).mean()) if len(top_indices) > 1 else 0.0
    _train_std_ref = _diversity_data.get("train_std")
    _topk_ratio = _topk_std / (_train_std_ref + 1e-8) if _train_std_ref else None
    _diversity_data["topk_std"] = _topk_std
    _diversity_data["topk_ratio"] = _topk_ratio
    if _topk_ratio is not None:
        print(f"[sample_cfg] top-k diversity: topk_std={_topk_std:.4f}  "
              f"topk_ratio={_topk_ratio:.3f}  "
              f"({'OK' if _topk_ratio > 0.7 else 'low' if _topk_ratio > 0.3 else 'COLLAPSE'})")

    w_tag = f"cfg_w{args.guidance_scale:.1f}_{args.sampler}"
    out_dir = Path(run_dir) / _diff_subdir / w_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "z_samples.npy", z_samples)

    import json
    with open(out_dir / "diversity.json", "w") as _f:
        json.dump(_diversity_data, _f)

    # t-SNE: train vs sampled vs top-k selected
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        if _train_latent_path.exists():
            _z_train_tsne = np.load(_train_latent_path)
            _n_train = len(_z_train_tsne)
            _combined = np.vstack([_z_train_tsne, z_samples])
            _perp = min(30, max(5, len(_combined) // 10))
            _emb = TSNE(n_components=2, perplexity=_perp, random_state=42,
                        n_iter=1000).fit_transform(_combined)
            _emb_train = _emb[:_n_train]
            _emb_all   = _emb[_n_train:]
            _emb_topk  = _emb_all[top_indices]

            _fig, _ax = _plt.subplots(figsize=(7, 6))
            _ax.scatter(_emb_train[:, 0], _emb_train[:, 1],
                        s=8, alpha=0.25, c="gray", label=f"train ({_n_train})")
            _ax.scatter(_emb_all[:, 0], _emb_all[:, 1],
                        s=8, alpha=0.35, c="steelblue", label=f"sampled ({len(z_samples)})")
            _ax.scatter(_emb_topk[:, 0], _emb_topk[:, 1],
                        s=25, alpha=0.9, c="tomato", label=f"top-{len(top_indices)} selected")
            _ax.set_title(f"t-SNE latent space — {w_tag}")
            _ax.legend(fontsize=8)
            _ax.grid(True, alpha=0.3)
            _fig.tight_layout()
            _fig.savefig(out_dir / "tsne.png", dpi=150, bbox_inches="tight")
            _plt.close(_fig)
            print(f"[sample_cfg] t-SNE → {out_dir / 'tsne.png'}")
    except Exception as _e:
        print(f"[sample_cfg] t-SNE skipped: {_e}")

    all_rows = []
    for pool_name, (z_pool, smiles_pool, y_lookup) in retrieval_pools.items():
        rows = []
        for idx in top_indices:
            pred_i = float(pred_batch[idx])
            for smi, sim in retrieve_nearest(z_samples[idx], z_pool, smiles_pool, top_k=5):
                rows.append({
                    "smiles": smi,
                    "pred_pIC50": pred_i,
                    "actual_pIC50": y_lookup.get(smi, float("nan")),
                    "cosine_sim": float(sim),
                    "source": pool_name,
                })

        _rows_df = pd.DataFrame(rows)
        n_retrieved = len(_rows_df)
        n_unique = _rows_df["smiles"].nunique()
        pool_df = (
            _rows_df
            .sort_values(["pred_pIC50", "cosine_sim"], ascending=[False, False])
            .drop_duplicates(subset=["smiles"], keep="first")
            .head(args.top_k)
            .reset_index(drop=True)
        )
        pool_df.to_csv(out_dir / f"top_candidates_{pool_name}.csv", index=False)
        all_rows.append(pool_df)
        print(f"[sample_cfg] {pool_name}: retrieved={n_retrieved}  unique={n_unique}  "
              f"final={len(pool_df)} → top_candidates_{pool_name}.csv")

    # Combined (all pools, deduplicated)
    candidates_df = (
        pd.concat(all_rows, ignore_index=True)
        .sort_values(["pred_pIC50", "cosine_sim"], ascending=[False, False])
        .drop_duplicates(subset=["smiles"], keep="first")
        .head(args.top_k)
        .reset_index(drop=True)
    )
    candidates_df.to_csv(out_dir / "top_candidates.csv", index=False)

    best_pred = float(pred_batch.max()) if len(pred_batch) else float("nan")
    print(f"\n[sample_cfg] best_pred_pIC50={best_pred:.4f}")
    print(f"[sample_cfg] saved → {out_dir}/")
    print(candidates_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
