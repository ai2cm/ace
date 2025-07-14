from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .activations import CappedGELU


class BilinearUpsample(torch.nn.Module):
    def __init__(self, upsampling: int = 2, **kwargs):
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=upsampling, mode="bilinear")

    def forward(self, x):
        return self.upsampler(x)


class AvgPool(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(pooling)

    def forward(self, x):
        return self.avgpool(x)


class ConvNeXtBlock(torch.nn.Module):
    """
    A convolution block as reported in https://github.com/CognitiveModeling/dlwp-hpx/blob/main/src/dlwp-hpx/dlwp/model/modules/blocks.py.
    This is a modified version of the actual ConvNextblock which
    is used in the HealPix paper.
    """

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: torch.nn.Module = CappedGELU,
        pad: str = "circular",
        norm: str | None = "instance",
        norm_kwargs: Mapping[str, Any] | None = None,
        upscale_factor: int = 4,
    ):
        super().__init__()
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"

        self.N_in = in_channels
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = pad
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        if self.norm_kwargs is None:
            self.norm_kwargs = {}

        assert n_layers == 1, "Can only use a single layer here!"  # Needs fixing

        # 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )

        # Convolution block
        convblock = []
        convblock.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        # Batch Norm
        if norm == "batch":
            convblock.append(
                torch.nn.BatchNorm2d(in_channels * upscale_factor, **self.norm_kwargs)
            )
        # Instance Norm
        elif norm == "instance":
            convblock.append(
                torch.nn.InstanceNorm2d(
                    in_channels * upscale_factor, **self.norm_kwargs
                )
            )
        # Layer Norm
        elif norm == "layer":
            convblock.append(
                torch.nn.LayerNorm(in_channels * upscale_factor, **self.norm_kwargs)
            )
        # No Norm
        elif norm is None:
            pass
        else:
            raise NotImplementedError(f"Normalization {norm} not implemented")

        convblock.append(activation())

        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        # Batch Norm
        if norm == "batch":
            convblock.append(
                torch.nn.BatchNorm2d(in_channels * upscale_factor, **self.norm_kwargs)
            )
        # Instance Norm
        elif norm == "instance":
            convblock.append(
                torch.nn.InstanceNorm2d(
                    in_channels * upscale_factor, **self.norm_kwargs
                )
            )
        # Layer Norm
        elif norm == "layer":
            convblock.append(
                torch.nn.LayerNorm(in_channels * upscale_factor, **self.norm_kwargs)
            )
        # No Norm
        elif norm is None:
            pass
        else:
            raise NotImplementedError(f"Normalization {norm} not implemented")

        convblock.append(activation())

        # Linear postprocessing
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        skip = self.skip_module(x)
        for layer in self.convblock:
            if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] != 1:
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            if isinstance(layer, torch.nn.LayerNorm):
                x = x.permute(0, 2, 3, 1).contiguous()
                x = layer(x)
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                x = layer(x)
        return skip + x
