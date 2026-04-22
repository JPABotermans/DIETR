"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch
import torchvision


class DIETRConvNext(torch.nn.Module):
    def __init__(
        self,
        size: str,
        freeze_first_layer: bool,
        freeze_bn: bool,
    ) -> None:
        super().__init__()
        if size == "S":
            self.backbone = torchvision.models.convnext.convnext_small(
                weights=torchvision.models.ConvNeXt_Small_Weights.DEFAULT
            )
        elif size == "T":
            self.backbone = torchvision.models.convnext.convnext_tiny(
                weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
            )
        else:
            ValueError(f"Backbone size needs to be [T]iny or [S]mall, not: {size}")
        
        if freeze_first_layer:
            for parameters in self.backbone.features[0].parameters():
                parameters.requires_grad = False

        self.layer1 = self.backbone.features[:2]
        self.layer2 = self.backbone.features[2:4]
        self.layer3 = self.backbone.features[4:6]
        self.layer4 = self.backbone.features[6:8]
        
        
        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input images of shape [B, 3, H, W]

        Returns:
            dict[str, torch.Tensor]: Multi-scale features.
            
            input = 	(8, 3, 640, 640)
            scale_2 -> 	features.shape=torch.Size([8, 96, 160, 160])
            scale_3 -> 	features.shape=torch.Size([8, 192, 80, 80])
            scale_4 -> 	features.shape=torch.Size([8, 384, 40, 40])
            scale_5 -> 	features.shape=torch.Size([8, 768, 20, 20])
        """
        features = {}

        x = self.layer1(x)
        features["scale_2"] = x

        x = self.layer2(x)
        features["scale_3"] = x

        x = self.layer3(x)
        features["scale_4"] = x

        x = self.layer4(x)
        features["scale_5"] = x

        return features
