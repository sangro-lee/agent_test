from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from tqdm import tqdm

from src.utils.io import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_graph: bool = False,
        early_stopping_patience: int = 15,
        checkpoint_every: int = 10,
        run_dir: Optional[str | Path] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.is_graph = is_graph
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_every = checkpoint_every
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.model.to(self.device)

    def _extract_xy(self, batch):
        if self.is_graph:
            batch = batch.to(self.device)
            y = batch.y.view(-1).to(self.device)
            return batch, y

        x, y = batch
        return x.to(self.device), y.to(self.device)

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in tqdm(loader, desc="Train", leave=False):
            x, y = self._extract_xy(batch)
            self.optimizer.zero_grad()
            pred, _ = self.model(x)
            loss = self.criterion(pred.view(-1), y.view(-1))
            loss.backward()
            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

        return total_loss / max(1, n_samples)

    @torch.no_grad()
    def eval_epoch(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        for batch in tqdm(loader, desc="Eval", leave=False):
            x, y = self._extract_xy(batch)
            pred, _ = self.model(x)
            loss = self.criterion(pred.view(-1), y.view(-1))

            bs = y.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

        return total_loss / max(1, n_samples)

    def fit(self, train_loader, val_loader, epochs: int) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_epoch = -1
        bad_epochs = 0

        ckpt_dir = None
        if self.run_dir is not None:
            ckpt_dir = self.run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_epoch = epoch
                bad_epochs = 0
                if ckpt_dir is not None:
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        best_val_loss=best_val,
                    )
            else:
                bad_epochs += 1

            if ckpt_dir is not None and self.checkpoint_every > 0 and epoch % self.checkpoint_every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_val_loss=best_val,
                )

            if bad_epochs >= self.early_stopping_patience:
                break

        if ckpt_dir is not None and (ckpt_dir / "best.pt").exists():
            state = torch.load(ckpt_dir / "best.pt", map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])

        history["best_val_loss"] = float(best_val)
        history["best_epoch"] = int(best_epoch)
        return history
