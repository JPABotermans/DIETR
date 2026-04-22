"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch
from dietr.modeling.base.blocks import  DownTopBlock, ConvBlock
from dietr.modeling.base.layers import ProjectionLayer


class DIETRMask(torch.nn.Module):
    def __init__(
        self,
        channels: dict[str, tuple[int, int]],
        n_prototypes: int,
        prj_channels: dict[str, tuple[int, int]] | None = None,
        latteral_conv: bool = False,
        latteral_conv_size: int = 256,
        latteral_conv_act: bool = False,
        latteral_conv_nrm: bool = True,
        prj_kernel_size: int = 3,
        prj_padding: int = 1,
        prj_stride: int = 1,
        prj_act: int | bool = True,
        prj_nrm: int | bool = True,
        
    ) -> None:
        super().__init__()

        # backbone feature projection
        self.prj_layer = ProjectionLayer(
            channels=prj_channels,
            kernel_size=prj_kernel_size,
            padding=prj_padding,
            stride=prj_stride,
            act=prj_act,
            nrm=prj_nrm,
        )

        self.down_top_block = DownTopBlock(
            channels=channels,
            latteral_conv=latteral_conv,
            latteral_conv_size=latteral_conv_size,
            latteral_conv_act=latteral_conv_act,
            latteral_conv_nrm=latteral_conv_nrm,        
        )
        self.final_conv = ConvBlock(
            in_channels=channels["scale_2"][1],
            out_channels=n_prototypes,
            kernel_size=3,
            padding=1,
            act=False,
            nrm=False,
        )
        

    def forward(self, backbone_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        backbone_features = self.prj_layer(backbone_features)
        return {"masks": self.final_conv(self.down_top_block(backbone_features)["scale_2"])}
    
