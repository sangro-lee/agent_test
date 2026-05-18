#!/usr/bin/env python
"""
GB-GA with per-sample latent-space fitness (Hungarian assignment).

Each molecule slot is assigned 1:1 to a z_sample via the Hungarian algorithm.
Each slot independently evolves toward its assigned z_target via crossover/mutation.

Fitness per slot = -||encode(mol) - z_target||²

Usage:
  python scripts/genetic_latent.py \
      --config    configs/experiments/rgcn_mlp_z4_random.yaml \
      --initpool  data/BACE1/bace1_clean_pic50.csv \
      --z_samples outputs/runs/MY_RUN/diffusion/.../z_samples.npy \
      --smiles_col smiles \
      --nstep 20 \
      --n_candidates 5 \
      --out_dir outputs/ga_latent
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import crossover as co
from src.features.fingerprints import smiles_to_fp
from src.models.mlp import FingerprintMLP
from src.utils.config import load_config
from src.utils.io import load_checkpoint


# ── encoder ────────────────────────────────────────────────────────────────

def build_encoder(run_dir: Path, feature_type: str, model_cfg: dict, feat_cfg: dict,
                  device: torch.device):
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
    return model, feature_type, is_graph


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


# ── Hungarian assignment ───────────────────────────────────────────────────

def hungarian_assign(initpool: list, z_samples: np.ndarray,
                     model, feat_cfg: dict, feature_type: str,
                     device: torch.device):
    """
    Encode initpool molecules and assign 1:1 to z_samples via Hungarian algorithm.
    If initpool is smaller than z_samples, samples with replacement to fill the gap.
    Returns (pool, assignment) where assignment[i] = z_sample index for slot i.
    """
    N = len(z_samples)

    if len(initpool) >= N:
        pool = random.sample(initpool, N)
    else:
        pool = list(initpool) + random.choices(initpool, k=N - len(initpool))

    print(f"[ga_latent] encoding {N} initpool molecules for Hungarian assignment...")
    z_init = encode(pool, model, feat_cfg, feature_type, device)  # (N, D)

    cost = cdist(z_init, z_samples, metric="sqeuclidean")  # (N, N)
    row_ind, col_ind = linear_sum_assignment(cost)
    # col_ind[i] = z_sample index assigned to slot i
    assignment = col_ind

    mean_l2 = float(np.mean(np.sqrt(cost[row_ind, col_ind])))
    print(f"[ga_latent] Hungarian done. initial mean_L2={mean_l2:.4f}")

    return pool, assignment


# ── per-slot distance ──────────────────────────────────────────────────────

def compute_dists(pool: list, assignment: np.ndarray, z_samples: np.ndarray,
                  model, feat_cfg: dict, feature_type: str,
                  device: torch.device) -> np.ndarray:
    """Returns squared L2 distance to assigned z_target for each slot."""
    N = len(pool)
    dists = np.full(N, 1e9, dtype=np.float32)

    valid_smiles, valid_idx = [], []
    for i, smi in enumerate(pool):
        if Chem.MolFromSmiles(smi) is not None:
            valid_smiles.append(smi)
            valid_idx.append(i)

    if not valid_smiles:
        return dists

    z = encode(valid_smiles, model, feat_cfg, feature_type, device)
    z_targets = z_samples[assignment[valid_idx]]
    sq = np.sum((z - z_targets) ** 2, axis=1)
    for k, i in enumerate(valid_idx):
        dists[i] = float(sq[k])

    return dists


# ── per-sample GB-GA step ──────────────────────────────────────────────────

def GB_GA_per_sample(
    model, feat_cfg: dict, feature_type: str, device: torch.device,
    pool: list, assignment: np.ndarray, z_samples: np.ndarray,
    current_dists: np.ndarray,
    istep: int, n_candidates: int, rxn_list: list, mu_prob: float,
    out_dir: Path,
):
    """
    One GA step: each slot generates n_candidates via crossover/mutation,
    batch-encodes them, and replaces its current molecule if a candidate
    is closer to the assigned z_target.
    """
    N = len(pool)
    z_targets = z_samples[assignment]  # (N, D)

    new_pool = list(pool)
    new_dists = current_dists.copy()

    # generate candidates for each slot
    all_slot_idx: list[int] = []
    all_cand_smis: list[str] = []

    for i in range(N):
        p_main = Chem.MolFromSmiles(pool[i])
        if p_main is None:
            continue

        generated = 0
        attempts = 0
        while generated < n_candidates and attempts < n_candidates * 10:
            attempts += 1

            p2 = Chem.MolFromSmiles(random.choice(pool))
            if p2 is None:
                continue

            child = co.crossover(p_main, p2)
            if child is None:
                continue

            if random.random() < mu_prob:
                cands_mut = []
                for rxn_sma in rxn_list:
                    rxn = AllChem.ReactionFromSmarts(rxn_sma)
                    try:
                        for mols in rxn.RunReactants((child,)):
                            if mols:
                                cands_mut.append(mols[0])
                    except Exception:
                        continue
                if not cands_mut:
                    continue
                child = np.random.choice(cands_mut)

            try:
                child_mol = Chem.MolFromSmiles(Chem.MolToSmiles(child), sanitize=True)
            except Exception:
                continue
            if child_mol is None:
                continue

            cano = Chem.MolToSmiles(child_mol, True)
            all_slot_idx.append(i)
            all_cand_smis.append(cano)
            generated += 1

    # batch encode all candidates and update slots
    if all_cand_smis:
        z_cands = encode(all_cand_smis, model, feat_cfg, feature_type, device)

        for k, (i, smi) in enumerate(zip(all_slot_idx, all_cand_smis)):
            dist = float(np.sum((z_cands[k] - z_targets[i]) ** 2))
            if dist < new_dists[i]:
                new_dists[i] = dist
                new_pool[i] = smi

    n_improved = int(np.sum(new_dists < current_dists))
    mean_l2 = float(np.mean(np.sqrt(new_dists)))

    print(f"step {istep+1:4d}  improved={n_improved}/{N}  mean_L2={mean_l2:.4f}  "
          f"n_cands={len(all_cand_smis)}", flush=True)

    if out_dir is not None:
        order = np.argsort(new_dists)
        lines = [
            f"slot={i}  z_idx={assignment[i]}  L2={np.sqrt(new_dists[i]):.4f}  smi={new_pool[i]}"
            for i in order[:50]
        ]
        (out_dir / f"INDEX_{istep+1}.dat").write_text("\n".join(lines))

    return new_pool, new_dists


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GB-GA with per-sample latent fitness (Hungarian assignment)"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--config",    help="YAML config path (auto-resolves run_dir)")
    grp.add_argument("--run_dir",   help="Run dir with checkpoints/best.pt and model_config.json")
    parser.add_argument("--feature_type", default=None,
                        help="fingerprint|graph|sme_graph (required when --run_dir used)")
    parser.add_argument("--initpool",     default="initpool.dat",
                        help="CSV/.dat with initial SMILES pool (default: initpool.dat)")
    parser.add_argument("--smiles_col",   default="SMILES")
    parser.add_argument("--z_samples",    required=True,
                        help="Path to z_samples.npy from sample_cfg.py")
    parser.add_argument("--nstep",        type=int,   default=20)
    parser.add_argument("--n_candidates", type=int,   default=5,
                        help="Crossover/mutation candidates per slot per step (default: 5)")
    parser.add_argument("--mu_prob",      type=float, default=0.3)
    parser.add_argument("--mutate_rxn",   default="mutate_reaction.dat")
    parser.add_argument("--out_dir",      default="outputs/ga_latent")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"[ga_latent] device={device}")

    if args.config:
        cfg          = load_config(args.config)
        feat_cfg     = cfg["features"]
        model_cfg    = cfg["model"]
        feature_type = str(feat_cfg.get("type", "fingerprint")).lower()
        from src.utils.io import resolve_run_dir
        run_dir = Path(resolve_run_dir(cfg, create_if_missing=False))
    else:
        run_dir = Path(args.run_dir)
        mc_path = run_dir / "model_config.json"
        model_cfg = json.loads(mc_path.read_text()) if mc_path.exists() else {}
        feature_type = args.feature_type or model_cfg.get("feature_type", "fingerprint")
        feat_cfg = model_cfg

    model, feature_type, _ = build_encoder(run_dir, feature_type, model_cfg, feat_cfg, device)
    print(f"[ga_latent] feature_type={feature_type}")

    z_samples = np.load(args.z_samples).astype(np.float32)
    print(f"[ga_latent] z_samples: {z_samples.shape}  (N={len(z_samples)}, D={z_samples.shape[1]})")

    import pandas as pd
    df = pd.read_csv(args.initpool)
    initpool = df[args.smiles_col].dropna().astype(str).tolist()
    print(f"[ga_latent] initpool: {len(initpool)} molecules")

    with open(args.mutate_rxn) as f:
        rxn_list = [l.strip() for l in f if l.strip()]

    co.average_size = 30
    co.size_stdev   = 40
    co.string_type  = "SMILES"

    # 1:1 Hungarian assignment: mol_i → z_samples[assignment[i]]
    pool, assignment = hungarian_assign(
        initpool, z_samples, model, feat_cfg, feature_type, device
    )
    current_dists = compute_dists(pool, assignment, z_samples, model, feat_cfg, feature_type, device)

    t0 = time.time()
    for istep in range(args.nstep):
        pool, current_dists = GB_GA_per_sample(
            model, feat_cfg, feature_type, device,
            pool, assignment, z_samples, current_dists,
            istep, args.n_candidates, rxn_list, args.mu_prob,
            out_dir,
        )
        print(f"  elapsed: {time.time() - t0:.1f}s", flush=True)

    # save results sorted by L2 distance
    order = np.argsort(current_dists)
    import pandas as pd
    result = pd.DataFrame({
        "smiles":       [pool[i]                         for i in order],
        "z_target_idx": [int(assignment[i])              for i in order],
        "l2_dist":      [float(np.sqrt(current_dists[i])) for i in order],
    })
    result.to_csv(out_dir / "final_pool.csv", index=False)

    print(f"\n[ga_latent] Done. {len(pool)} molecules → {out_dir}/final_pool.csv")
    print(f"[ga_latent] top-5 L2 dists: {np.sqrt(current_dists[order[:5]])}")


if __name__ == "__main__":
    main()
