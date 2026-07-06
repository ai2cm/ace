from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .activations import CappedGELU
from .utils import PartialDepthwiseConv2d

class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class MaskAwareInputBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
    ):
        super().__init__()

        self.partial_spatial = PartialDepthwiseConv2d(
            channels=in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        # We concatenate the partial-convolution features and masks.
        mixing_channels = 2 * in_channels
        expanded_channels = upscale_factor * in_channels

        self.norm = ChannelLayerNorm(mixing_channels)

        self.expand = nn.Conv2d(
            mixing_channels,
            expanded_channels,
            kernel_size=1,
        )
        self.activation = CappedGELU()
        self.contract = nn.Conv2d(
            expanded_channels,
            out_channels,
            kernel_size=1,
        )

        self.skip = nn.Conv2d(
            mixing_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor,
        ocean_column_mask: torch.Tensor,
    ) -> torch.Tensor:

        channel_mask = channel_mask.to(dtype=x.dtype)
        # channel_mask is stored as [1, C, H, W]; torch.cat requires an exact
        # batch-dim match, so expand it here before any concatenation.
        channel_mask = channel_mask.expand(x.shape[0], -1, -1, -1)
        masked_input = x * channel_mask

        spatial_features, _ = self.partial_spatial(
            masked_input,
            channel_mask,
        )

        # The network sees both the locally estimated values and which
        # original values were genuinely present.
        features = torch.cat(
            [spatial_features, channel_mask],
            dim=1,
        )

        residual_input = torch.cat(
            [masked_input, channel_mask],
            dim=1,
        )

        y = self.norm(features)
        y = self.expand(y)
        y = self.activation(y)
        y = self.contract(y)

        output = self.skip(residual_input) + y

        # Latent features are only valid over actual ocean columns.
        return output * ocean_column_mask

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
        checkpoint_strategy: Literal["all", "simple"] | None = None,
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
        self.checkpoint_strategy = checkpoint_strategy
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

    def _apply_simple_checkpoint(self, layer, x):
        if self.checkpoint_strategy == "simple" and not isinstance(layer, nn.Conv2d):
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            x = layer(x)
        return x

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
                x = self._apply_simple_checkpoint(layer, x)
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                x = self._apply_simple_checkpoint(layer, x)
        return skip + x
