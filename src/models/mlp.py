from typing import List

import torch
from torch import nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}


class FingerprintMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        activation: str = "relu",
        no_latent_act: bool = False,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")

        act_cls = _ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            if not (no_latent_act and i == len(hidden_dims) - 1):
                layers.extend([act_cls(), nn.Dropout(dropout)])
            prev = h

        self.encoder = nn.Sequential(*layers)
        self.out_layer = nn.Linear(prev, 1)
        self._latent_dim = prev

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        pred = self.out_layer(latent).view(-1)
        return pred, latent

    def get_latent_dim(self) -> int:
        return self._latent_dim
