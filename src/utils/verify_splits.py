"""
Compare split index files between two run directories.

Usage:
    python -m src.utils.verify_splits <dir_a> <dir_b>
    python -m src.utils.verify_splits outputs/runs/run_a outputs/runs/run_b
"""

import sys
from pathlib import Path

import numpy as np


def compare_splits(dir_a: str | Path, dir_b: str | Path) -> bool:
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    splits = ["train_idx", "val_idx", "test_idx"]
    all_match = True

    print(f"Comparing splits:\n  A: {dir_a}\n  B: {dir_b}\n")

    for name in splits:
        path_a = dir_a / "splits" / f"{name}.npy"
        path_b = dir_b / "splits" / f"{name}.npy"

        if not path_a.exists():
            print(f"  [MISSING] {path_a}")
            all_match = False
            continue
        if not path_b.exists():
            print(f"  [MISSING] {path_b}")
            all_match = False
            continue

        a = np.load(path_a)
        b = np.load(path_b)

        set_a, set_b = set(a.tolist()), set(b.tolist())
        only_a = len(set_a - set_b)
        only_b = len(set_b - set_a)
        common = len(set_a & set_b)

        if set_a == set_b:
            print(f"  {name}: IDENTICAL  (n={len(a)})")
        else:
            all_match = False
            print(f"  {name}: DIFFER  "
                  f"common={common}  only_A={only_a}  only_B={only_b}  "
                  f"|A|={len(a)}  |B|={len(b)}")

    print()
    if all_match:
        print("Result: all splits are identical.")
    else:
        print("Result: splits differ.")

    return all_match


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.utils.verify_splits <dir_a> <dir_b>")
        sys.exit(1)
    ok = compare_splits(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
