#!/usr/bin/env python
"""
GB-GA with two fitness modes:

  pred    — fitness = encoder's predicted pIC50 (global ranking, original approach)
  latent  — fitness = -||encode(mol) - z_target||² with 1:1 Hungarian assignment

Usage:
  # pred mode (no z_samples needed)
  python scripts/genetic_latent.py --config configs/... --mode pred

  # latent mode
  python scripts/genetic_latent.py --config configs/... --mode latent \
      --z_samples outputs/runs/.../z_samples.npy
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
    """Returns latent z only."""
    if feature_type == "fingerprint":
        bits   = int(feat_cfg.get("fp_bits", 2048))
        radius = int(feat_cfg.get("fp_radius", 2))
        chiral = bool(feat_cfg.get("use_chirality", False))
        fps = smiles_to_fp(smiles_list, bits=bits, radius=radius, use_chirality=chiral)
        x   = torch.tensor(fps, dtype=torch.float32, device=device)
        _, z = model(x)
        return z.cpu().numpy()

    else:
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


@torch.no_grad()
def encode_with_pred(smiles_list: list, model, feat_cfg: dict,
                     feature_type: str, device: torch.device):
    """Returns (pred_pIC50, z) both as np.ndarray."""
    if feature_type == "fingerprint":
        bits   = int(feat_cfg.get("fp_bits", 2048))
        radius = int(feat_cfg.get("fp_radius", 2))
        chiral = bool(feat_cfg.get("use_chirality", False))
        fps = smiles_to_fp(smiles_list, bits=bits, radius=radius, use_chirality=chiral)
        x   = torch.tensor(fps, dtype=torch.float32, device=device)
        pred, z = model(x)
        return pred.cpu().numpy().flatten(), z.cpu().numpy()

    else:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        if feature_type == "sme_graph":
            from src.features.graph import SMEMolDataset
            smask_type = str(feat_cfg.get("smask_type", "brics"))
            ds = SMEMolDataset(smiles_list, [0.0] * len(smiles_list), smask_type=smask_type)
        else:
            from src.features.graph import MolDataset
            ds = MolDataset(smiles_list, [0.0] * len(smiles_list))
        loader = PyGDataLoader(ds, batch_size=256, shuffle=False)
        preds, zs = [], []
        for batch in loader:
            batch = batch.to(device)
            pred, z = model(batch)
            preds.append(pred.detach().cpu().numpy().flatten())
            zs.append(z.detach().cpu().numpy())
        return np.concatenate(preds), np.concatenate(zs, axis=0)


# ── shared crossover/mutation helper ──────────────────────────────────────

def _generate_child(p_main, pool: list, rxn_list: list, mu_prob: float):
    """Returns canonical SMILES of a valid child, or None."""
    p2 = Chem.MolFromSmiles(random.choice(pool))
    if p2 is None:
        return None

    child = co.crossover(p_main, p2)
    if child is None:
        return None

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
            return None
        child = np.random.choice(cands_mut)

    try:
        child_mol = Chem.MolFromSmiles(Chem.MolToSmiles(child), sanitize=True)
    except Exception:
        return None
    if child_mol is None:
        return None

    return Chem.MolToSmiles(child_mol, True)


# ── pred mode ──────────────────────────────────────────────────────────────

def fitness_pred(smiles_list: list, model, feat_cfg: dict,
                 feature_type: str, device: torch.device) -> np.ndarray:
    """Returns predicted pIC50 for each SMILES (invalid → -1e9)."""
    valid_smiles, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        if Chem.MolFromSmiles(smi) is not None:
            valid_smiles.append(smi)
            valid_idx.append(i)

    scores = np.full(len(smiles_list), -1e9, dtype=np.float32)
    if not valid_smiles:
        return scores

    preds, _ = encode_with_pred(valid_smiles, model, feat_cfg, feature_type, device)
    for k, i in enumerate(valid_idx):
        scores[i] = float(preds[k])
    return scores


def GB_GA_pred(
    model, feat_cfg: dict, feature_type: str, device: torch.device,
    GenPool: list, istep: int,
    gau_sigma: float, target_pool: int, rxn_list: list, mu_prob: float,
    out_dir: Path,
):
    """Global-ranking GA with pred_pIC50 fitness."""
    scores = fitness_pred(GenPool, model, feat_cfg, feature_type, device)

    sort_a = sorted(scores)
    cut = sort_a[int(len(sort_a) * 0.2)] if sort_a else -1e9

    x_parents1, mol_list = [], []

    if istep == 0:
        x_parents1 = list(GenPool)
    else:
        for imol, smi in enumerate(GenPool):
            target = np.random.normal(0.0, gau_sigma)
            diff   = abs(target - scores[imol])
            if diff < (cut - sort_a[0]):
                x_parents1.append(smi)
                mol_list.append(scores[imol])
            elif random.random() >= 0.8:
                x_parents1.append(smi)

        log_msg = "\n".join(
            f"  {smi}  pred_pIC50={scores[i]:.4f}"
            for i, smi in enumerate(GenPool) if scores[i] >= cut
        )
        (out_dir / f"INDEX_{istep}.dat").write_text(log_msg)

    mean_val = float(np.mean(mol_list)) if mol_list else float("nan")
    std_val  = float(np.std(mol_list))  if mol_list else float("nan")

    x_parents2 = []
    ncross, nmut = 0, 0
    need_offspring = max(0, target_pool - len(x_parents1))

    while ncross < need_offspring:
        p1 = Chem.MolFromSmiles(random.choice(x_parents1))
        if p1 is None:
            continue
        cano = _generate_child(p1, x_parents1, rxn_list, mu_prob)
        if cano is None:
            continue

        child_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(cano), 3, 2048)
        sim_thresh = max(0.5, 0.8 - istep * 0.01)
        too_similar = any(
            Chem.MolFromSmiles(s) is not None and
            AllChem.DataStructs.TanimotoSimilarity(
                child_fp,
                AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 3, 2048)
            ) > sim_thresh
            for pool in (x_parents1, x_parents2) for s in pool
        )
        if too_similar:
            continue

        ncross += 1
        x_parents2.append(cano)

    GenPool = list(dict.fromkeys(
        Chem.MolToSmiles(Chem.MolFromSmiles(s), True)
        for s in x_parents1 + x_parents2
        if Chem.MolFromSmiles(s) is not None
    ))

    if istep != 0:
        print(f"step {istep+1:4d}  cut={cut:.4f}  mean={mean_val:.4f}  std={std_val:.4f}"
              f"  pool={len(GenPool)} ({len(x_parents1)}+{len(x_parents2)})"
              f"  cross={ncross}", flush=True)

    return GenPool


# ── latent mode ────────────────────────────────────────────────────────────

def hungarian_assign(initpool: list, z_samples: np.ndarray,
                     model, feat_cfg: dict, feature_type: str,
                     device: torch.device):
    N = len(z_samples)
    if len(initpool) >= N:
        pool = random.sample(initpool, N)
    else:
        pool = list(initpool) + random.choices(initpool, k=N - len(initpool))

    print(f"[ga_latent] encoding {N} initpool molecules for Hungarian assignment...")
    z_init = encode(pool, model, feat_cfg, feature_type, device)

    cost = cdist(z_init, z_samples, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = col_ind

    mean_l2 = float(np.mean(np.sqrt(cost[row_ind, col_ind])))
    print(f"[ga_latent] Hungarian done. initial mean_L2={mean_l2:.4f}")
    return pool, assignment


def compute_dists(pool: list, assignment: np.ndarray, z_samples: np.ndarray,
                  model, feat_cfg: dict, feature_type: str,
                  device: torch.device) -> np.ndarray:
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


def GB_GA_latent(
    model, feat_cfg: dict, feature_type: str, device: torch.device,
    pool: list, assignment: np.ndarray, z_samples: np.ndarray,
    current_dists: np.ndarray,
    istep: int, n_candidates: int, rxn_list: list, mu_prob: float,
    out_dir: Path,
):
    """Per-slot GA: each slot evolves independently toward its assigned z_target."""
    N = len(pool)
    z_targets = z_samples[assignment]

    new_pool = list(pool)
    new_dists = current_dists.copy()

    all_slot_idx: list[int] = []
    all_cand_smis: list[str] = []

    for i in range(N):
        p_main = Chem.MolFromSmiles(pool[i])
        if p_main is None:
            continue
        generated, attempts = 0, 0
        while generated < n_candidates and attempts < n_candidates * 10:
            attempts += 1
            cano = _generate_child(p_main, pool, rxn_list, mu_prob)
            if cano is None:
                continue
            all_slot_idx.append(i)
            all_cand_smis.append(cano)
            generated += 1

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
        description="GB-GA: pred (pIC50 fitness) or latent (Hungarian z-target fitness)"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--config",    help="YAML config path (auto-resolves run_dir)")
    grp.add_argument("--run_dir",   help="Run dir with checkpoints/best.pt and model_config.json")
    parser.add_argument("--feature_type", default=None,
                        help="fingerprint|graph|sme_graph (required when --run_dir used)")
    parser.add_argument("--mode",         default="latent", choices=["pred", "latent"],
                        help="pred: pIC50 fitness  |  latent: per-sample z-target fitness (default: latent)")
    parser.add_argument("--initpool",     default="initpool.dat",
                        help="CSV/.dat with initial SMILES pool (default: initpool.dat)")
    parser.add_argument("--smiles_col",   default="SMILES")
    parser.add_argument("--z_samples",    default=None,
                        help="z_samples.npy from sample_cfg.py (required for --mode latent)")
    parser.add_argument("--nstep",        type=int,   default=20)
    # latent mode
    parser.add_argument("--n_candidates", type=int,   default=5,
                        help="[latent] candidates per slot per step (default: 5)")
    # pred mode
    parser.add_argument("--target_pool",  type=int,   default=400,
                        help="[pred] target pool size (default: 400)")
    parser.add_argument("--gau_sigma",    type=float, default=0.001,
                        help="[pred] selection noise in fitness space (default: 0.001)")
    parser.add_argument("--mu_prob",      type=float, default=0.3)
    parser.add_argument("--mutate_rxn",   default="mutate_reaction.dat")
    parser.add_argument("--out_dir",      default="outputs/ga_latent")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    if args.mode == "latent" and args.z_samples is None:
        parser.error("--z_samples is required for --mode latent")

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"[ga_latent] device={device}  mode={args.mode}")

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

    import pandas as pd
    df = pd.read_csv(args.initpool)
    initpool = df[args.smiles_col].dropna().astype(str).tolist()
    print(f"[ga_latent] initpool: {len(initpool)} molecules")

    with open(args.mutate_rxn) as f:
        rxn_list = [l.strip() for l in f if l.strip()]

    co.average_size = 30
    co.size_stdev   = 40
    co.string_type  = "SMILES"

    t0 = time.time()

    # ── pred mode ─────────────────────────────────────────────────────────
    if args.mode == "pred":
        GenPool = list(initpool)
        for istep in range(args.nstep):
            GenPool = GB_GA_pred(
                model, feat_cfg, feature_type, device,
                GenPool, istep,
                args.gau_sigma, args.target_pool, rxn_list, args.mu_prob,
                out_dir,
            )
            print(f"  elapsed: {time.time() - t0:.1f}s", flush=True)

        final_scores = fitness_pred(GenPool, model, feat_cfg, feature_type, device)
        order = np.argsort(-final_scores)
        result = pd.DataFrame({
            "smiles":      [GenPool[i]           for i in order],
            "pred_pIC50":  [float(final_scores[i]) for i in order],
        })
        result.to_csv(out_dir / "final_pool.csv", index=False)
        print(f"\n[ga_latent] Done. {len(GenPool)} molecules → {out_dir}/final_pool.csv")
        print(f"[ga_latent] top-5 pred_pIC50: {final_scores[order[:5]]}")

    # ── latent mode ───────────────────────────────────────────────────────
    else:
        z_samples = np.load(args.z_samples).astype(np.float32)
        print(f"[ga_latent] z_samples: {z_samples.shape}")

        pool, assignment = hungarian_assign(
            initpool, z_samples, model, feat_cfg, feature_type, device
        )
        current_dists = compute_dists(
            pool, assignment, z_samples, model, feat_cfg, feature_type, device
        )

        for istep in range(args.nstep):
            pool, current_dists = GB_GA_latent(
                model, feat_cfg, feature_type, device,
                pool, assignment, z_samples, current_dists,
                istep, args.n_candidates, rxn_list, args.mu_prob,
                out_dir,
            )
            print(f"  elapsed: {time.time() - t0:.1f}s", flush=True)

        order = np.argsort(current_dists)
        result = pd.DataFrame({
            "smiles":       [pool[i]                          for i in order],
            "z_target_idx": [int(assignment[i])               for i in order],
            "l2_dist":      [float(np.sqrt(current_dists[i])) for i in order],
        })
        result.to_csv(out_dir / "final_pool.csv", index=False)
        print(f"\n[ga_latent] Done. {len(pool)} molecules → {out_dir}/final_pool.csv")
        print(f"[ga_latent] top-5 L2 dists: {np.sqrt(current_dists[order[:5]])}")


if __name__ == "__main__":
    main()
