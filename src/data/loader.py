from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_activity_csv(
    csv_path: str,
    smiles_col: str,
    activity_col: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data CSV not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in [smiles_col, activity_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    report = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_missing_smiles": int(df[smiles_col].isna().sum()),
        "n_missing_activity": int(df[activity_col].isna().sum()),
    }
    return df, report
