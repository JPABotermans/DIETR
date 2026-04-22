"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch

def get_act(act: str | None | bool, **kwargs) -> torch.nn.Module:
    return {
        "silu": torch.nn.SiLU,
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "gelu": torch.nn.GELU,
        "None": torch.nn.Identity,
        None: torch.nn.Identity,
        False: torch.nn.Identity,
        True: torch.nn.ReLU,
    }[act](**kwargs)


def get_nrm(nrm: str | None | bool, **kwargs) -> torch.nn.Module:
    return {
        "Batch2D": torch.nn.BatchNorm2d,
        "Group": torch.nn.GroupNorm,
        None: torch.nn.Identity,
        "None": torch.nn.Identity,
        False: torch.nn.Identity,
        True: torch.nn.BatchNorm2d
    }[nrm](**kwargs)



class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        act: str  = "relu",
        act_kwargs: dict[str, any] = {}
    ):
        super().__init__()
        self.n_layers = n_layers
        h = [hidden_dim] * (n_layers - 1)
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(
                n,
                k,
            )
            for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_act(act, **act_kwargs)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.n_layers - 1 else layer(x)
        return x



