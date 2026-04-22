"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch
from dietr.modeling.base.blocks import ConvBlock


class ProjectionLayer(torch.nn.Module):
    def __init__(
        self,
        channels: dict[str, tuple[int, int]],
        kernel_size: int,
        padding: int,
        stride: int,
        act: int,
        nrm: int,
    ) -> None:
        """Projects all input scales while not doing any resampling.
        is defined by the channels: dict["scale_2", [in_channels, out_channels]]
        """
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                scale: ConvBlock(
                    in_channels=channels[scale][0],
                    out_channels=channels[scale][1],
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    act=act,
                    nrm=nrm,
                )
                for scale in channels.keys()
            }
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {scale: layer(x[scale]) for scale, layer in self.layers.items()}
