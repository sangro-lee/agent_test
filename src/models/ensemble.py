from typing import Iterable

import torch
from torch import nn


class EnsembleModel(nn.Module):
    def __init__(self, models: Iterable[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(list(models))
        if len(self.models) == 0:
            raise ValueError("EnsembleModel requires at least one model.")

    def forward(self, batch):
        preds = []
        latents = []
        for model in self.models:
            pred, latent = model(batch)
            preds.append(pred)
            latents.append(latent)

        pred_mean = torch.stack(preds, dim=0).mean(dim=0)
        latent_concat = torch.cat(latents, dim=-1)
        return pred_mean, latent_concat
