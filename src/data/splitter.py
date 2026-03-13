from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _compute_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("Invalid split fractions.")

    scaffold_groups = defaultdict(list)
    for idx, smi in enumerate(df[smiles_col].astype(str).tolist()):
        scaffold_groups[_compute_scaffold(smi)].append(idx)

    rng = np.random.default_rng(seed)
    groups: List[List[int]] = list(scaffold_groups.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n_total = len(df)
    n_test_target = int(round(n_total * test_frac))
    n_val_target = int(round(n_total * val_frac))

    train_idx, val_idx, test_idx = [], [], []

    for g in groups:
        if len(test_idx) + len(g) <= n_test_target:
            test_idx.extend(g)
        elif len(val_idx) + len(g) <= n_val_target:
            val_idx.extend(g)
        else:
            train_idx.extend(g)

    # If split got skewed by large scaffold blocks, top up val/test from train.
    train_arr = np.array(train_idx, dtype=int)
    rng.shuffle(train_arr)

    def _top_up(target: List[int], target_n: int, pool: np.ndarray) -> np.ndarray:
        need = max(0, target_n - len(target))
        if need == 0:
            return pool
        take = pool[:need].tolist()
        target.extend(take)
        return pool[need:]

    train_arr = _top_up(test_idx, n_test_target, train_arr)
    train_arr = _top_up(val_idx, n_val_target, train_arr)

    train_idx = train_arr.tolist()

    return (
        np.array(sorted(train_idx), dtype=int),
        np.array(sorted(val_idx), dtype=int),
        np.array(sorted(test_idx), dtype=int),
    )


def random_split(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("Invalid split fractions.")

    n = len(df)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    return (
        np.array(sorted(train_idx), dtype=int),
        np.array(sorted(val_idx), dtype=int),
        np.array(sorted(test_idx), dtype=int),
    )
