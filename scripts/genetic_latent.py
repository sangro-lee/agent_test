#!/usr/bin/env python
"""
GB-GA with latent-space KDE fitness.

Fitness = KDE log-density of encode(mol) under the distribution of
diffusion-sampled latent vectors (z_samples).

Usage:
  python scripts/genetic_latent.py \
      --config    configs/experiments/mlp_random.yaml \
      --initpool  data/initpool.csv \
      --z_samples outputs/runs/MY_RUN/diffusion/.../z_samples.npy \
      --smiles_col smiles \
      --nstep 10 \
      --target_pool 400 \
      --out_dir outputs/ga_latent
"""
import argparse
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.neighbors import KernelDensity

import crossover as co
from src.features.fingerprints import smiles_to_fp
from src.models.mlp import FingerprintMLP
from src.utils.config import load_config
from src.utils.io import load_checkpoint


# ── encoder ────────────────────────────────────────────────────────────────

def build_encoder(cfg: dict, device: torch.device):
    """Build and load encoder. Returns (model, feat_cfg, feature_type, is_graph).
    Supports feature types: fingerprint, graph (AttentiveFP), sme_graph (SME-RGCN)."""
    feat_cfg     = cfg["features"]
    model_cfg    = cfg["model"]
    feature_type = str(feat_cfg.get("type", "fingerprint")).lower()

    from src.utils.io import resolve_run_dir
    run_dir = resolve_run_dir(cfg, create_if_missing=False)

    if feature_type == "fingerprint":
        probe = smiles_to_fp(
            ["CCO"],
            bits=int(feat_cfg.get("fp_bits", 2048)),
            radius=int(feat_cfg.get("fp_radius", 2)),
            use_chirality=bool(feat_cfg.get("use_chirality", False)),
        )
        model = FingerprintMLP(
            input_dim=probe.shape[1],
            hidden_dims=list(model_cfg.get("hidden_dims", [512, 4])),
            dropout=float(model_cfg.get("dropout", 0.0)),
            activation=str(model_cfg.get("activation", "relu")),
            no_latent_act=bool(model_cfg.get("no_latent_act", False)),
        ).to(device)
        is_graph = False

    elif feature_type == "graph":
        from src.models.gnn import AttentiveFPModel
        model = AttentiveFPModel(
            in_channels=10, edge_dim=6,
            hidden_dim=int(model_cfg.get("gnn_hidden", 200)),
            num_layers=int(model_cfg.get("gnn_layers", 3)),
            dropout=float(model_cfg.get("gnn_dropout", 0.1)),
        ).to(device)
        is_graph = True

    elif feature_type == "sme_graph":
        from src.models.gnn import SMERGCNModel
        from src.features.graph import SME_NODE_DIM
        _ffn_dims_cfg = model_cfg.get("sme_ffn_dims")
        if _ffn_dims_cfg is not None:
            _latent_dim = int(model_cfg.get("latent_dim", 4))
            sme_kwargs = {"ffn_dims": [int(d) for d in _ffn_dims_cfg] + [_latent_dim]}
        else:
            sme_kwargs = {"ffn_hidden": int(model_cfg.get("sme_ffn_hidden", 200))}
        model = SMERGCNModel(
            in_feats=SME_NODE_DIM,
            hidden_feats=list(model_cfg.get("sme_hidden_feats", [200, 200])),
            rgcn_dropout=float(model_cfg.get("sme_rgcn_dropout", 0.25)),
            ffn_dropout=float(model_cfg.get("sme_ffn_dropout", 0.25)),
            **sme_kwargs,
        ).to(device)
        is_graph = True

    else:
        raise ValueError(f"Unsupported features.type: {feature_type}")

    load_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, map_location=str(device))
    model.eval()
    return model, feat_cfg, feature_type, is_graph


@torch.no_grad()
def encode(smiles_list: list, model, feat_cfg: dict,
           feature_type: str, device: torch.device) -> np.ndarray:
    if feature_type == "fingerprint":
        bits   = int(feat_cfg.get("fp_bits", 2048))
        radius = int(feat_cfg.get("fp_radius", 2))
        chiral = bool(feat_cfg.get("use_chirality", False))
        fps = smiles_to_fp(smiles_list, bits=bits, radius=radius, use_chirality=chiral)
        x   = torch.tensor(fps, dtype=torch.float32, device=device)
        _, z = model(x)
        return z.cpu().numpy()

    else:  # graph / sme_graph
        from torch_geometric.loader import DataLoader as PyGDataLoader
        if feature_type == "sme_graph":
            from src.features.graph import SMEMolDataset
            smask_type = str(feat_cfg.get("smask_type", "brics"))
            ds = SMEMolDataset(smiles_list, [0.0] * len(smiles_list), smask_type=smask_type)
        else:
            from src.features.graph import MolDataset
            ds = MolDataset(smiles_list, [0.0] * len(smiles_list))
        loader = PyGDataLoader(ds, batch_size=256, shuffle=False)
        zs = []
        for batch in loader:
            batch = batch.to(device)
            _, z = model(batch)
            zs.append(z.detach().cpu().numpy())
        return np.concatenate(zs, axis=0)


# ── KDE fitness ────────────────────────────────────────────────────────────

def build_kde(z_samples: np.ndarray, bandwidth: str | float = "scott") -> KernelDensity:
    """Fit KDE on z_samples. bandwidth='scott' uses Scott's rule (auto)."""
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(z_samples)
    return kde


def fitness_kde(smiles_list: list, model, feat_cfg: dict, feature_type: str,
                kde: KernelDensity, device: torch.device) -> np.ndarray:
    """Return KDE log-density for each SMILES (higher = better)."""
    valid, indices = [], []
    for i, smi in enumerate(smiles_list):
        if Chem.MolFromSmiles(smi) is not None:
            valid.append(smi)
            indices.append(i)

    scores = np.full(len(smiles_list), -1e9)
    if not valid:
        return scores

    z = encode(valid, model, feat_cfg, feature_type, device)
    log_dens = kde.score_samples(z)
    for i, idx in enumerate(indices):
        scores[idx] = float(log_dens[i])
    return scores


# ── GB-GA cycle ────────────────────────────────────────────────────────────

def GB_GA(
    model, feat_cfg, feature_type, kde, device,
    GenPool, istep,
    gau_sigma, target_pool, rxn_list, mu_prob,
    out_dir: Path,
):
    scores = fitness_kde(GenPool, model, feat_cfg, feature_type, kde, device)

    sort_a = sorted(scores)
    cut = sort_a[int(len(sort_a) * 0.2)] if sort_a else -1e9

    x_parents1 = []
    mol_list   = []

    if istep == 0:
        x_parents1 = list(GenPool)
    else:
        for imol, smi in enumerate(GenPool):
            target = np.random.normal(0.0, gau_sigma)   # noise around 0 in log-density space
            diff   = abs(target - scores[imol])
            if diff < (cut - sort_a[0]):                 # closer to best density → keep
                x_parents1.append(smi)
                mol_list.append(scores[imol])
            elif random.random() >= 0.8:
                x_parents1.append(smi)

        log_msg = "\n".join(
            f"  {smi}  score={scores[i]:.4f}"
            for i, smi in enumerate(GenPool) if scores[i] >= cut
        )
        (out_dir / f"INDEX_{istep}.dat").write_text(log_msg)

    mean_val = float(np.mean(mol_list)) if mol_list else float("nan")
    std_val  = float(np.std(mol_list))  if mol_list else float("nan")

    # ── Crossover & Mutation ───────────────────────────────────────────────
    x_parents2 = []
    ncross, nmut = 0, 0
    need_offspring = max(0, target_pool - len(x_parents1))

    while ncross < need_offspring:
        p1 = Chem.MolFromSmiles(random.choice(x_parents1))
        p2 = Chem.MolFromSmiles(random.choice(x_parents1))
        child = co.crossover(p1, p2)
        if child is None:
            continue

        l_mut = False
        if random.random() < mu_prob:
            l_mut = True
            candidates = []
            for rxn_sma in rxn_list:
                rxn = AllChem.ReactionFromSmarts(rxn_sma)
                for mols in rxn.RunReactants((child,)):
                    candidates.append(mols[0])
            if not candidates:
                continue
            child = np.random.choice(candidates)

        child = Chem.MolFromSmiles(Chem.MolToSmiles(child), sanitize=True)
        if child is None:
            continue

        cano = Chem.MolToSmiles(child, True)
        child_fp = AllChem.GetMorganFingerprintAsBitVect(child, 3, 2048)

        # similarity dedup (relaxes over steps)
        sim_thresh = max(0.5, 0.8 - istep * 0.01)
        too_similar = False
        for pool in (x_parents1, x_parents2):
            for smi in pool:
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048)
                if DataStructs.TanimotoSimilarity(child_fp, fp) > sim_thresh:
                    too_similar = True
                    break
            if too_similar:
                break
        if too_similar:
            continue

        ncross += 1
        if l_mut:
            nmut += 1
        x_parents2.append(cano)

    GenPool = list(dict.fromkeys(
        Chem.MolToSmiles(Chem.MolFromSmiles(s), True)
        for s in x_parents1 + x_parents2
        if Chem.MolFromSmiles(s) is not None
    ))

    if istep != 0:
        print(f"{istep+1:6d}  cut={cut:.4f}  mean={mean_val:.4f}  std={std_val:.4f}"
              f"  pool={len(GenPool)} ({len(x_parents1)}+{len(x_parents2)})"
              f"  cross={ncross}  mut={nmut}", flush=True)

    return GenPool


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--initpool",    required=True, help="CSV with initial SMILES pool")
    parser.add_argument("--smiles_col",  default="smiles")
    parser.add_argument("--z_samples",   required=True, help="Path to z_samples.npy from sample_cfg.py")
    parser.add_argument("--nstep",       type=int,   default=10)
    parser.add_argument("--target_pool", type=int,   default=400)
    parser.add_argument("--mu_prob",     type=float, default=0.3)
    parser.add_argument("--gau_sigma",   type=float, default=0.001)
    parser.add_argument("--bandwidth",   default="scott",
                        help="KDE bandwidth: 'scott', 'silverman', or float (default: scott)")
    parser.add_argument("--mutate_rxn",  default="mutate_reaction.dat")
    parser.add_argument("--out_dir",     default="outputs/ga_latent")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"[ga_latent] device={device}")

    cfg = load_config(args.config)
    model, feat_cfg, feature_type, _ = build_encoder(cfg, device)
    print(f"[ga_latent] feature_type={feature_type}")

    z_samples = np.load(args.z_samples).astype(np.float32)
    print(f"[ga_latent] z_samples: {z_samples.shape}")

    bw = args.bandwidth if args.bandwidth in ("scott", "silverman") else float(args.bandwidth)
    kde = build_kde(z_samples, bandwidth=bw)
    print(f"[ga_latent] KDE fitted  bandwidth={bw}")

    import pandas as pd
    df = pd.read_csv(args.initpool)
    GenPool = df[args.smiles_col].dropna().astype(str).tolist()
    print(f"[ga_latent] initpool: {len(GenPool)} molecules")

    with open(args.mutate_rxn) as f:
        rxn_list = [l.strip() for l in f if l.strip()]

    co.average_size = 30
    co.size_stdev   = 40
    co.string_type  = "SMILES"

    t0 = time.time()
    for istep in range(args.nstep):
        GenPool = GB_GA(
            model, feat_cfg, feature_type, kde, device,
            GenPool, istep,
            args.gau_sigma, args.target_pool, rxn_list, args.mu_prob,
            out_dir,
        )
        print(f"  step {istep+1} elapsed: {time.time()-t0:.1f}s", flush=True)

    # save final pool with scores
    final_scores = fitness_kde(GenPool, model, feat_cfg, feature_type, kde, device)
    order = np.argsort(-final_scores)
    import pandas as pd
    result = pd.DataFrame({
        "smiles":    [GenPool[i] for i in order],
        "kde_score": [final_scores[i] for i in order],
    })
    result.to_csv(out_dir / "final_pool.csv", index=False)
    print(f"\n[ga_latent] Done. {len(GenPool)} molecules → {out_dir}/final_pool.csv")
    print(f"[ga_latent] top-5 KDE scores: {final_scores[order[:5]]}")


if __name__ == "__main__":
    main()
