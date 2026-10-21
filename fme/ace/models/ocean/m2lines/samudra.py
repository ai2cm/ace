import dataclasses
import functools
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, get_args

import numpy as np
import torch
import torch.nn as nn

from fme.ace.models.ocean.m2lines.layers import (
    AvgPool,
    BilinearUpsample,
    ConvNeXtBlock,
    LatPad,
    ZonallyPeriodicBilinearUpsample,
    pad_latitude,
)
from fme.ace.models.ocean.m2lines.utils import pairwise
from fme.core.models.conditional_sfno.layers import Context, ContextConfig

ConditionedBlocks = Literal["bottleneck", "all_blocks"]


def validate_pooling_factors(pooling: Sequence[int], n_levels: int) -> None:
    """Raise ValueError unless ``pooling`` holds one positive integer per level."""
    if len(pooling) != n_levels:
        raise ValueError(
            f"pooling has {len(pooling)} factors, expected one per level "
            f"({n_levels}, the length of ch_width)"
        )
    for factor in pooling:
        if factor < 1:
            raise ValueError(
                f"pooling factors must be positive integers, got {list(pooling)}"
            )


def validate_exact_pooling_grid(
    pooling: Sequence[int], img_shape: tuple[int, int], lat_pad: LatPad
) -> None:
    """Raise ValueError unless every level of ``pooling`` divides ``img_shape``
    exactly, and, under ``lat_pad="pole"``, every level's width is even so the
    pole continuation rotates by exactly 180 degrees of longitude.
    """
    height, width = img_shape
    product = math.prod(pooling)
    if height % product != 0 or width % product != 0:
        raise ValueError(
            f"the product of pooling factors {list(pooling)} is {product}, which "
            f"does not divide the grid shape {tuple(img_shape)}"
        )
    if lat_pad == "pole":
        level_width = width
        for factor in pooling:
            level_width //= factor
            if level_width % 2 != 0:
                raise ValueError(
                    f'lat_pad="pole" needs an even width at every level, but '
                    f"pooling {list(pooling)} on width {width} gives width "
                    f"{level_width}"
                )


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
    zonally_periodic_upsample : bool, optional
        If True, use bilinear upsampling that enforces periodicity along the
        longitude axis in the decoder, removing the lon=0 seam introduced by the
        default (non-periodic) bilinear upsampling. By default False to preserve
        the behavior of checkpoints trained without it.
    lat_pad : {"constant", "pole"}, optional
        Latitude padding used wherever the network pads latitude (see
        ``pad_latitude``), except that the upsampler ignores it when
        ``zonally_periodic_upsample`` is False and, under "constant", the
        periodic upsampler does not pad latitude. By default "constant", the
        original behavior. Adds no parameters, so checkpoints load across modes.
    context_config : ContextConfig, optional
        If given (with a non-zero noise embedding), the ConvNeXt blocks selected
        by ``conditioned_blocks`` take a conditional scale and bias off the noise
        field in the ``Context`` passed to ``forward``. Only noise conditioning
        is supported so far; scalar, label, and positional embeddings are not.
    conditioned_blocks : {"bottleneck", "all_blocks"}, optional
        Which ConvNeXt blocks are conditioned. ``"bottleneck"`` conditions only
        the block at the coarsest resolution. ``"all_blocks"`` conditions every block.
        Required when ``context_config`` is given.
    pooling : List[int], optional
        Pooling factor of each level of the U-Net, applied to both axes, one
        per entry of ``ch_width``; each decoder level upsamples by the factor
        its encoder level pooled by. A factor of 1 keeps that level's
        resolution. When given, the product of the factors must divide both
        grid dimensions, so every level pools and upsamples exactly, with no
        row or column dropped or refilled; ``forward`` raises otherwise. Under
        ``lat_pad="pole"`` every level's width must also be even. By default
        None: a factor of 2 at every level, floor-pooling grids that 2 does not
        divide and refilling the dropped rows and columns in the decoder.
        Adds no parameters, so checkpoints load across settings at equal
        ``ch_width``.

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
        zonally_periodic_upsample: bool = False,
        context_config: ContextConfig | None = None,
        conditioned_blocks: ConditionedBlocks | None = None,
        lat_pad: LatPad = "constant",
        pooling: list[int] | None = None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
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
        self.zonally_periodic_upsample = zonally_periodic_upsample
        if lat_pad not in get_args(LatPad):
            raise ValueError(f"unknown lat_pad {lat_pad!r}")
        self.lat_pad = lat_pad
        self.exact_pooling = pooling is not None
        if pooling is None:
            self.pooling = [2] * len(ch_width)
        else:
            validate_pooling_factors(pooling, len(ch_width))
            self.pooling = list(pooling)
        upsample_cls = (
            functools.partial(ZonallyPeriodicBilinearUpsample, lat_pad=lat_pad)
            if zonally_periodic_upsample
            else BilinearUpsample
        )

        if context_config is not None:
            if context_config.embed_dim_noise <= 0:
                raise ValueError(
                    "Samudra conditioning is noise-only, so context_config must "
                    "have embed_dim_noise > 0."
                )
            if (
                context_config.embed_dim_scalar > 0
                or context_config.embed_dim_labels > 0
                or context_config.embed_dim_pos > 0
            ):
                raise ValueError(
                    "Samudra only supports noise conditioning; scalar, label, and "
                    "positional embeddings are not implemented."
                )
            if conditioned_blocks is None:
                raise ValueError("context_config requires conditioned_blocks to be set")
        elif conditioned_blocks is not None:
            raise ValueError("conditioned_blocks requires context_config to be set")
        self.conditioned_blocks = conditioned_blocks

        # Called once per block in construction order: num_steps encoder
        # blocks, the bottleneck, then num_steps decoder blocks.
        num_steps = len(self.ch_width)
        n_built = 0

        def block_context() -> ContextConfig | None:
            nonlocal n_built
            is_bottleneck = n_built == num_steps
            n_built += 1
            if conditioned_blocks == "all_blocks":
                return context_config
            if conditioned_blocks == "bottleneck" and is_bottleneck:
                return context_config
            return None

        ch_width_with_input = (self.input_channels, *self.ch_width)

        # going down
        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width_with_input)):
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
                    context_config=block_context(),
                    lat_pad=self.lat_pad,
                )
            )
            layers.append(AvgPool(pooling=self.pooling[i]))
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
                context_config=block_context(),
                lat_pad=self.lat_pad,
            )
        )
        pooling_reversed = self.pooling[::-1]
        layers.append(
            upsample_cls(in_channels=b, out_channels=b, upsampling=pooling_reversed[0])
        )
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
                    context_config=block_context(),
                    lat_pad=self.lat_pad,
                )
            )
            layers.append(
                upsample_cls(
                    in_channels=b, out_channels=b, upsampling=pooling_reversed[i + 1]
                )
            )
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
                context_config=block_context(),
                lat_pad=self.lat_pad,
            )
        )
        layers.append(torch.nn.Conv2d(b, self.output_channels, self.last_kernel_size))

        if n_built != 2 * num_steps + 1:
            raise AssertionError(
                f"built {n_built} ConvNeXt blocks, expected {2 * num_steps + 1}"
            )

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width_with_input) - 1)

    def forward(self, fts, context: Context | None = None):
        if self.exact_pooling:
            validate_exact_pooling_grid(
                self.pooling, (fts.shape[-2], fts.shape[-1]), self.lat_pad
            )
        temp: list[torch.Tensor] = []
        count = 0
        for layer in self.layers:
            crop = fts.shape[2:]
            if isinstance(layer, nn.Conv2d):
                fts = pad_latitude(fts, self.N_pad, self.N_pad, self.lat_pad)
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
            # only the ConvNeXt blocks are conditionable; the pooling, upsample
            # and final conv layers take the tensor alone
            if isinstance(layer, ConvNeXtBlock):
                layer_args: tuple = (fts, context)
            else:
                layer_args = (fts,)
            if self.checkpoint_strategy == "all":
                fts = torch.utils.checkpoint.checkpoint(
                    layer, *layer_args, use_reentrant=False
                )
            else:
                fts = layer(*layer_args)
            if count < self.num_steps:
                if isinstance(layer, ConvNeXtBlock):
                    temp.append(fts)
                    count += 1
            elif count >= self.num_steps:
                if isinstance(
                    layer, BilinearUpsample | ZonallyPeriodicBilinearUpsample
                ):
                    skip = temp[int(2 * self.num_steps - count - 1)]
                    if self.exact_pooling:
                        if fts.shape != skip.shape:
                            raise ValueError(
                                f"upsampled shape {tuple(fts.shape)} differs from "
                                f"the skip connection's {tuple(skip.shape)} under "
                                f"exact pooling {self.pooling}"
                            )
                    else:
                        crop = np.array(fts.shape[2:])
                        shape = np.array(skip.shape[2:])
                        pads = shape - crop
                        pads_lr = (pads[1] // 2, pads[1] - pads[1] // 2, 0, 0)
                        fts = pad_latitude(
                            fts, pads[0] // 2, pads[0] - pads[0] // 2, self.lat_pad
                        )
                        fts = nn.functional.pad(fts, pads_lr, mode=self.pad)
                    fts += skip
                    count += 1
        return fts
