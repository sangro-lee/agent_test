import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_run_dir(config: Dict[str, Any]) -> Path:
    output_cfg = config.get("output", {})
    run_root = output_cfg.get("run_root", "auto")

    if run_root != "auto":
        run_dir = Path(run_root)
    else:
        base_dir = Path(output_cfg.get("base_dir", "outputs/runs"))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / ts

    ensure_dir(run_dir)
    return run_dir


def resolve_run_dir(config: Dict[str, Any], create_if_missing: bool = False) -> Path:
    output_cfg = config.get("output", {})
    run_root = output_cfg.get("run_root", "auto")

    if run_root != "auto":
        run_dir = Path(run_root)
        if create_if_missing:
            ensure_dir(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    base_dir = Path(output_cfg.get("base_dir", "outputs/runs"))
    if create_if_missing:
        ensure_dir(base_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / ts
        ensure_dir(run_dir)
        return run_dir

    if not base_dir.exists():
        raise FileNotFoundError(
            f"No base run directory found: {base_dir}. Run preprocess first or set output.run_root."
        )
    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No existing runs under {base_dir}. Run preprocess first or set output.run_root."
        )
    return sorted(candidates)[-1]


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        ckpt.update(extra)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)


def load_checkpoint(
    path: str | Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def save_numpy(path: str | Path, arr: np.ndarray):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)


def load_numpy(path: str | Path) -> np.ndarray:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"NumPy file not found: {in_path}")
    return np.load(in_path)


def save_json(path: str | Path, data: Dict[str, Any]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"JSON file not found: {in_path}")
    with in_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_predictions_csv(path: str | Path, df: pd.DataFrame):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
