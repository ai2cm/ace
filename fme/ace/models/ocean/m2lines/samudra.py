import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from fme.ace.models.ocean.m2lines.layers import (
    AvgPool,
    BilinearUpsample,
    ConvNeXtBlock,
    MaskAwareInputBlock,
    MaskedAvgPool2d,
)
from fme.ace.models.ocean.m2lines.utils import pairwise
from fme.core.dataset_info import DatasetInfo


class Samudra(torch.nn.Module):
    """
    Samudra Network from M2Lines.

    Parameters
    ----------
    input_channels : int
        Number of input channels, including forcing variables and history
    output_channels : int
        Number of output channels in the final layer
    ch_width : List[int]
        Channel widths for each level of the U-Net architecture
    dilation : List[int]
        Dilation rates for each ConvNeXt block
    n_layers : List[int]
        Number of ConvNeXt layers at each level
    pad : str, optional
        Type of padding to use in convolutions, for example,
        ('circular', 'constant'), by default "circular"
    norm: str, optional
        Normalization to use in the network, by default "instance"
        Options are "batch", "layer", "instance", or None
        "layer" normalization normalizes over only the channel dimensions
    partial_convolutions : bool
        Whether to use partial convolutions and masking, by default False

    Example:
    --------
    >>> import torch
    >>> from fme.ace.models.ocean.m2lines.samudra import Samudra
    >>> model = Samudra(
    ...     input_channels=4,
    ...     output_channels=3,
    ...     ch_width=[8],
    ...     dilation=[2],
    ...     n_layers=[1],
    ... )
    >>> model(torch.randn(1, 4, 128, 128)).shape
    torch.Size([1, 3, 128, 128])
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dataset_info: DatasetInfo | None = None,
        in_names: list[str] | None = None,
        ch_width: list[int] = dataclasses.field(
            default_factory=lambda: [200, 250, 300, 400]
        ),
        dilation: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8]),
        n_layers: list[int] = dataclasses.field(default_factory=lambda: [1, 1, 1, 1]),
        pad: str = "circular",
        norm: str | None = "instance",
        norm_kwargs: Mapping[str, Any] | None = None,
        upscale_factor: int = 4,
        checkpoint_strategy: Literal["all", "simple"] | None = None,
        partial_convolutions: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dataset_info = dataset_info
        self.hist = 0  # Fixed
        self.ch_width = ch_width
        self.dilation = dilation
        self.n_layers = n_layers
        self.pad = pad
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.last_kernel_size = 3
        self.N_pad = int((self.last_kernel_size - 1) / 2)
        self.upscale_factor = upscale_factor
        self.checkpoint_strategy = checkpoint_strategy
        self.partial_convolutions = partial_convolutions

        ch_width_with_input = (self.input_channels, *self.ch_width)

        Conv0: type[MaskAwareInputBlock] | type[ConvNeXtBlock]
        Pool: type[MaskedAvgPool2d] | type[AvgPool]
        if self.partial_convolutions:
            Conv0 = MaskAwareInputBlock
            Pool = MaskedAvgPool2d
            # Build per-channel mask [1, C, H, W] and 2D ocean column mask [1, 1, H, W].
            # When dataset_info / in_names are not provided (e.g. in unit tests) we fall
            # back to all-ones, which disables masking without changing forward.
            if dataset_info is not None:
                spatial_mask_provider = dataset_info.spatial_mask_provider
                img_shape = dataset_info.img_shape  # (H, W)

                if in_names is not None:
                    per_channel_masks = []
                    for name in in_names:
                        m = spatial_mask_provider.get_mask_tensor_for(name)
                        if m is None:
                            m = torch.ones(img_shape)
                        per_channel_masks.append(m)
                    channel_mask = torch.stack(per_channel_masks, dim=0).unsqueeze(0)
                else:
                    channel_mask = torch.ones(1, input_channels, *img_shape)

                ocean_col = spatial_mask_provider.get_mask_tensor_for("mask_2d")
                if ocean_col is None:
                    ocean_col = torch.ones(img_shape)
                ocean_column_mask = ocean_col.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            else:
                # No spatial information available — masks are constructed lazily in
                # forward() from the first input tensor's shape.
                channel_mask = None
                ocean_column_mask = None

            # Buffers may be None when dataset_info is absent; forward handles this.
            if channel_mask is not None:
                self.register_buffer("channel_mask", channel_mask)
            else:
                self.channel_mask: torch.Tensor | None = None  # type: ignore[assignment]
            if ocean_column_mask is not None:
                self.register_buffer("ocean_column_mask", ocean_column_mask)
            else:
                self.ocean_column_mask: torch.Tensor | None = None  # type: ignore[assignment]
        else:
            Conv0 = ConvNeXtBlock
            Pool = AvgPool
        # going down
        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width_with_input)):
            if i == 0:
                layers.append(
                    Conv0(
                        a,
                        b,
                        dilation=self.dilation[i],
                        n_layers=self.n_layers[i],
                        pad=self.pad,
                        norm=self.norm,
                        norm_kwargs=self.norm_kwargs,
                        upscale_factor=self.upscale_factor,
                        checkpoint_strategy=self.checkpoint_strategy,
                    )
                )
            else:
                layers.append(
                    ConvNeXtBlock(
                        a,
                        b,
                        dilation=self.dilation[i],
                        n_layers=self.n_layers[i],
                        pad=self.pad,
                        norm=self.norm,
                        norm_kwargs=self.norm_kwargs,
                        upscale_factor=self.upscale_factor,
                        checkpoint_strategy=self.checkpoint_strategy,
                    )
                )
            layers.append(Pool())
        layers.append(
            ConvNeXtBlock(
                b,
                b,
                dilation=self.dilation[i],
                n_layers=self.n_layers[i],
                pad=self.pad,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                upscale_factor=self.upscale_factor,
                checkpoint_strategy=self.checkpoint_strategy,
            )
        )
        layers.append(BilinearUpsample())
        ch_width_with_input_reversed = ch_width_with_input[::-1]
        dilation_reversed = self.dilation[::-1]
        n_layers_reversed = self.n_layers[::-1]
        for i, (a, b) in enumerate(pairwise(ch_width_with_input_reversed[:-1])):
            layers.append(
                ConvNeXtBlock(
                    a,
                    b,
                    dilation=dilation_reversed[i],
                    n_layers=n_layers_reversed[i],
                    pad=self.pad,
                    norm=self.norm,
                    norm_kwargs=self.norm_kwargs,
                    upscale_factor=self.upscale_factor,
                    checkpoint_strategy=self.checkpoint_strategy,
                )
            )
            layers.append(BilinearUpsample())
        layers.append(
            ConvNeXtBlock(
                b,
                b,
                dilation=dilation_reversed[i],
                n_layers=n_layers_reversed[i],
                pad=self.pad,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                upscale_factor=self.upscale_factor,
                checkpoint_strategy=self.checkpoint_strategy,
            )
        )
        layers.append(torch.nn.Conv2d(b, self.output_channels, self.last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width_with_input) - 1)

    def _get_fallback_masks(
        self, fts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all-ones masks matching the spatial shape of fts.

        Used when dataset_info was not provided (e.g. unit tests without masks).
        """
        B, C, H, W = fts.shape
        channel_mask = torch.ones(1, C, H, W, device=fts.device, dtype=fts.dtype)
        ocean_column_mask = torch.ones(1, 1, H, W, device=fts.device, dtype=fts.dtype)
        return channel_mask, ocean_column_mask

    def forward(self, fts):
        if self.partial_convolutions:
            # Resolve masks — use registered buffers when available, otherwise fall back
            # to all-ones (no masking) tensors derived from the input shape.
            if self.channel_mask is not None:
                channel_mask = self.channel_mask
                current_mask = self.ocean_column_mask  # [1, 1, H, W]
            else:
                channel_mask, current_mask = self._get_fallback_masks(fts)

        temp: list[torch.Tensor] = []
        count = 0
        for layer in self.layers:
            if isinstance(layer, MaskAwareInputBlock):
                fts = layer(
                    fts,
                    channel_mask=channel_mask,
                    ocean_column_mask=current_mask,
                )
                # MaskAwareInputBlock replaces the i=0 ConvNeXtBlock; save its
                # output as the first encoder skip connection.
                if count < self.num_steps:
                    temp.append(fts)
                    count += 1

            elif isinstance(layer, MaskedAvgPool2d):
                # current_mask [1, 1, H, W] broadcasts against fts [B, C, H, W].
                # The returned output_mask is [1, 1, H/2, W/2] — ready for the
                # next pooling stage without any reshaping.
                fts, current_mask = layer(fts, current_mask)

            elif isinstance(layer, AvgPool):
                fts = layer(fts)

            elif isinstance(layer, BilinearUpsample):
                if self.checkpoint_strategy == "all":
                    fts = torch.utils.checkpoint.checkpoint(
                        layer, fts, use_reentrant=False
                    )
                else:
                    fts = layer(fts)
                if count >= self.num_steps:
                    # U-Net skip connection: pad/crop fts to match the encoder
                    # skip tensor if spatial dims differ (e.g. odd input sizes).
                    skip_idx = int(2 * self.num_steps - count - 1)
                    crop = np.array(fts.shape[2:])
                    shape = np.array(temp[skip_idx].shape[2:])
                    pads = shape - crop
                    pads_lr = (pads[1] // 2, pads[1] - pads[1] // 2, 0, 0)
                    pads_tb = (0, 0, pads[0] // 2, pads[0] - pads[0] // 2)
                    fts = nn.functional.pad(fts, pads_lr, mode=self.pad)
                    fts = nn.functional.pad(fts, pads_tb, mode="constant")
                    fts += temp[skip_idx]
                    count += 1

            elif isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
                if self.checkpoint_strategy == "all":
                    fts = torch.utils.checkpoint.checkpoint(
                        layer, fts, use_reentrant=False
                    )
                else:
                    fts = layer(fts)

            else:  # ConvNeXtBlock
                if self.checkpoint_strategy == "all":
                    fts = torch.utils.checkpoint.checkpoint(
                        layer, fts, use_reentrant=False
                    )
                else:
                    fts = layer(fts)
                if count < self.num_steps:
                    temp.append(fts)
                    count += 1

        return fts
