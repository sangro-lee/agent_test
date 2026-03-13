from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem


_UNIT_TO_MOLAR = {
    "m": 1.0,
    "molar": 1.0,
    "mol": 1.0,
    "m/l": 1.0,
    "mm": 1e-3,
    "millimolar": 1e-3,
    "mmol": 1e-3,
    "um": 1e-6,
    "μm": 1e-6,
    "micromolar": 1e-6,
    "umol": 1e-6,
    "nm": 1e-9,
    "nanomolar": 1e-9,
    "nmol": 1e-9,
    "pm": 1e-12,
    "picomolar": 1e-12,
    "pmol": 1e-12,
}


def sanitize_smiles(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    work = df.copy()
    work[smiles_col] = work[smiles_col].astype(str).str.strip()

    mols = work[smiles_col].apply(lambda s: Chem.MolFromSmiles(s) if s else None)
    valid_mask = mols.notna()
    clean = work.loc[valid_mask].copy()
    clean[smiles_col] = clean[smiles_col].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
    )
    clean.reset_index(drop=True, inplace=True)
    return clean


def _normalize_unit(unit: str) -> str:
    return unit.strip().lower().replace(" ", "")


def convert_to_pIC50(
    df: pd.DataFrame,
    activity_col: str,
    activity_type: str,
    unit_col: Optional[str] = None,
) -> pd.Series:
    act_type = activity_type.strip().upper()
    values = pd.to_numeric(df[activity_col], errors="coerce")

    if act_type == "PIC50":
        return values.astype(float)

    if act_type not in {"IC50", "KI", "KD"}:
        raise ValueError(f"Unsupported activity_type: {activity_type}")

    if unit_col is None or unit_col not in df.columns:
        raise ValueError(
            f"unit column is required for {activity_type} conversion but missing: {unit_col}"
        )

    units = df[unit_col].astype(str).map(_normalize_unit)
    factors = units.map(_UNIT_TO_MOLAR)
    molar = values * factors

    with np.errstate(divide="ignore", invalid="ignore"):
        pic50 = -np.log10(molar.astype(float))

    return pd.Series(pic50, index=df.index, dtype=float)


def clean_activity_dataframe(
    df: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
    activity_type: str,
    unit_col: Optional[str] = None,
    clip_min: float = 2.0,
    clip_max: float = 12.0,
) -> pd.DataFrame:
    work = sanitize_smiles(df, smiles_col)
    work["pIC50"] = convert_to_pIC50(work, activity_col, activity_type, unit_col=unit_col)
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["pIC50"]).copy()
    work["pIC50"] = work["pIC50"].clip(lower=clip_min, upper=clip_max)
    work.reset_index(drop=True, inplace=True)
    return work
