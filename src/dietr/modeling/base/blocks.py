"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
"""
import torch
from dietr.modeling.base.projections import get_nrm, get_act


class ConvBlock(torch.nn.Module):
    """Default CNN block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        act: bool = True,
        nrm: bool = True,
        act_kwargs: dict[str, any] = {},
    ) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=not nrm,
        )
        self.nrm = get_nrm(nrm=nrm, num_features=out_channels)
        self.act = get_act(act=act, **act_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.nrm(x)
        return self.act(x)


class ResidualConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        act: str = "silu",
        act_kwargs: dict[str, any] = {},
    ) -> None:
        super().__init__()
        self.conv_block1 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
        )
        self.conv_block2 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            act=False,
        )
        self.act = get_act(act=act, **act_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.act(x + residual)


class ResidualBottleNeckBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        downsample_factor: int = 4,
        act: str = "silu",
        act_kwargs: dict[str, any] = {},
    ) -> None:
        super().__init__()
        self.conv_block1 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels // downsample_factor,
            kernel_size=1,
            padding=0,
        )
        self.conv_block2 = ConvBlock(
            in_channels=in_channels // downsample_factor,
            out_channels=in_channels // downsample_factor,
        )
        self.conv_block3 = ConvBlock(
            in_channels=in_channels // downsample_factor,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            act=False,
        )
        self.act = get_act(act=act, **act_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.act(x + residual)


class ResidualLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 3,
        expansion: float = 1.0,
        act: str = "silu",
        resample_fn: torch.nn.Module = torch.nn.Upsample(scale_factor=2.0),
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv_in = ConvBlock(in_channels, hidden_channels, act=act)
        self.residual_blocks = torch.nn.Sequential(
            *[
                ResidualBottleNeckBlock(hidden_channels, act=act)
                for _ in range(n_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv_out = ConvBlock(
                hidden_channels,
                out_channels,
                act=act,
            )
        else:
            self.conv_out = torch.nn.Identity()
        self.resample_fn = resample_fn

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_in(x)
        x = self.residual_blocks(x)
        output = self.conv_out(x)
        return output, self.resample_fn(output)


class UpBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.conv_in = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.blocks = torch.nn.Sequential(
            *[ResidualConvBlock(out_channels) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Projects the input tensor and then upsamples the ouptut.
        returns:
            x: projected input
            carry: the interpolated output, which can be futher processed in the next sale.
        """
        x = self.conv_in(x)
        x = self.blocks(x)
        return x, torch.nn.functional.interpolate(input=x, scale_factor=2)


class DownTopBlock(torch.nn.Module):
    def __init__(
        self,
        channels: dict[str, tuple[int, int]],
        block: torch.nn.Module = UpBlock,
        block_cfg: dict[str, any] = {},
        down_top: bool = True,
        latteral_conv: bool = False,
        latteral_conv_size: int = 256,
        latteral_conv_nrm: bool = False,
        latteral_conv_act: bool = False,
    ) -> None:
        super().__init__()
        self.latteral_conv = latteral_conv
        self.scales = sorted(channels.keys())
        self.layers = torch.nn.ModuleDict(
            {
                scale: block(
                    in_channels=in_channels, out_channels=out_channels, **block_cfg
                )
                for scale, (in_channels, out_channels) in channels.items()
            }
        )
        if self.latteral_conv:
            self.lateral_conv_dict = torch.nn.ModuleDict(
                {
                    scale: ConvBlock(
                        latteral_conv_size,
                        latteral_conv_size,
                        kernel_size=1,
                        padding=0,
                        nrm=latteral_conv_nrm,
                        act=latteral_conv_act,
                    )
                    for scale, (_, _) in channels.items()
                }
            )

        self.down_top = down_top

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Proceses the input.
            For each scale it:
                1. Concatenates the scale input (in x) with the carry of the previous layer.
                2. Processes the scale input which becomes the output for this scale.
                3. Resamples the scale ouput which becomes the carry input for the next scale.

        ToDo: make sure there is no resmapling for the last layer, because it will adds processing
        time we don't need
        Returns:
            outputs: dict[str, torch.Tensor]
                A dictionary where the keys are: "scale_2", "scale_3" etc, and the values are the
                output values for each scale.
        """
        scales_reversed = self.scales[::-1] if self.down_top else self.scales
        outputs = {}
        carry = x[self.scales[-1]] if self.down_top else x[self.scales[0]]
        for scale in scales_reversed:
            if self.latteral_conv:
                layer_input = self.lateral_conv_dict[scale](x[scale])
            else:
                layer_input = x[scale]
            layer_input = torch.concat([carry, layer_input], dim=1)
            hidden, carry = self.layers[scale](layer_input)
            outputs[scale] = hidden

        return outputs
