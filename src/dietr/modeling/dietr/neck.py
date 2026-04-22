"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch

from dietr.modeling.base.layers import ProjectionLayer
from dietr.modeling.base.blocks import ResidualLayer, DownTopBlock


class DIETRNeck(torch.nn.Module):
    """Base Neck, Feature pyramid network to refine features from multiple scales.
    """
    def __init__(
        self,
        channels: dict[str, tuple[int, int]],
        n_blocks: int,
        cnv_act: str ,
        expansion: float,
        latteral_conv: bool,
        latteral_conv_size: int,
        latteral_conv_act: bool,
        latteral_conv_nrm: bool,
        prj_kernel_size: int,
        prj_padding: int,
        prj_stride: int,
        prj_act: str | bool,
        prj_nrm: str | bool):
        super().__init__()
        hidden_dim = list(channels.values())[0][1]
        self.hidden_dim = hidden_dim
        fpn_channels = {k: [2 * hidden_dim, hidden_dim] for k in channels.keys()}

        self.prj_layer = ProjectionLayer(
            channels=channels,
            kernel_size=prj_kernel_size,
            padding=prj_padding,
            stride=prj_stride,
            act=prj_act,
            nrm=prj_nrm,
        )
        
        # top-down fpn
        self.fpn_down = DownTopBlock(
            channels=fpn_channels,
            block=ResidualLayer,
            block_cfg={
                "act": cnv_act,
                "expansion": expansion,
                "resample_fn": torch.nn.Upsample(scale_factor=2.0),
                "n_blocks": n_blocks,
            },
            latteral_conv=latteral_conv,
            latteral_conv_size=latteral_conv_size,
            latteral_conv_act=latteral_conv_act,
            latteral_conv_nrm=latteral_conv_nrm,
            
        )
        # Down-top fpn
        self.fpn_up = DownTopBlock(
            channels=fpn_channels,
            block=ResidualLayer,
            block_cfg={
                "act": cnv_act,
                "expansion": expansion,
                "resample_fn": torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                "n_blocks": n_blocks,
            },
            latteral_conv=latteral_conv,
            latteral_conv_size=latteral_conv_size,
            latteral_conv_act=latteral_conv_act,
            latteral_conv_nrm=latteral_conv_nrm,
            down_top=False,
        )



    def forward(
        self, backbone_features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Given backbone features, outputs refined features on multiple scales.

        Args:
            backbone_features (dict[str, torch.Tensor]): Features processed by the backbone, key is the scale and the values are the features.

        Returns:
            dict[str, torch.Tensor]: For each feature scale, the refined features. 
        """
        features = self.prj_layer(backbone_features)

        return self.fpn_up(self.fpn_down(features))
