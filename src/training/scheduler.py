from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer


def get_scheduler(optimizer: Optimizer, config: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    tr_cfg = config.get("training", {})
    scheduler_type = str(tr_cfg.get("scheduler", "cosine")).lower()

    if scheduler_type == "none":
        return None

    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(tr_cfg.get("lr_reduce_factor", 0.5)),
            patience=int(tr_cfg.get("lr_patience", 5)),
            min_lr=float(tr_cfg.get("min_lr", 1e-7)),
        )

    epochs = int(tr_cfg.get("epochs", 100))
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(tr_cfg.get("min_lr", 1e-7)),
    )
