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
    ZonallyPeriodicBilinearUpsample,
)
from fme.ace.models.ocean.m2lines.utils import pairwise
from fme.core.models.conditional_sfno.layers import Context, ContextConfig

NoiseInjection = Literal["bottleneck", "all_blocks"]


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
    context_config : ContextConfig, optional
        If given (with a non-zero noise embedding), the ConvNeXt blocks selected
        by ``noise_injection`` take a conditional scale and bias off the noise
        field in the ``Context`` passed to ``forward``. Only noise conditioning
        is supported; scalar, label, and positional embeddings are not.
    noise_injection : {"bottleneck", "all_blocks"}, optional
        Which ConvNeXt blocks are noise-conditioned. ``"bottleneck"`` conditions
        only the block at the coarsest resolution (DLESyM-Ocean's choice, which
        perturbs large scales); ``"all_blocks"`` conditions every block (the
        pattern the ACE SFNO uses). Required when ``context_config`` is given.

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
        noise_injection: NoiseInjection | None = None,
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
        upsample_cls = (
            ZonallyPeriodicBilinearUpsample
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
            if noise_injection is None:
                raise ValueError("context_config requires noise_injection to be set")
        elif noise_injection is not None:
            raise ValueError("noise_injection requires context_config to be set")
        self.noise_injection = noise_injection

        # Blocks are built in this order: the num_steps encoder blocks, the
        # bottleneck block, then the num_steps decoder blocks. `block_context`
        # is called once per block, in that order, and hands back the context
        # for the blocks this injection variant conditions.
        num_steps = len(self.ch_width)
        n_built = 0

        def block_context() -> ContextConfig | None:
            nonlocal n_built
            is_bottleneck = n_built == num_steps
            n_built += 1
            if noise_injection == "all_blocks":
                return context_config
            if noise_injection == "bottleneck" and is_bottleneck:
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
                )
            )
            layers.append(AvgPool())
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
            )
        )
        layers.append(upsample_cls(in_channels=b, out_channels=b))
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
                )
            )
            layers.append(upsample_cls(in_channels=b, out_channels=b))
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
            )
        )
        layers.append(torch.nn.Conv2d(b, self.output_channels, self.last_kernel_size))

        if n_built != 2 * num_steps + 1:
            # a bare assert would vanish under `python -O`, and miscounting
            # silently moves which block "bottleneck" conditions
            raise AssertionError(
                f"built {n_built} ConvNeXt blocks, expected {2 * num_steps + 1}"
            )

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width_with_input) - 1)

    def forward(self, fts, context: Context | None = None):
        temp: list[torch.Tensor] = []
        count = 0
        for layer in self.layers:
            crop = fts.shape[2:]
            if isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
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
                    crop = np.array(fts.shape[2:])
                    shape = np.array(
                        temp[int(2 * self.num_steps - count - 1)].shape[2:]
                    )
                    pads = shape - crop
                    pads_lr = (pads[1] // 2, pads[1] - pads[1] // 2, 0, 0)
                    pads_tb = (0, 0, pads[0] // 2, pads[0] - pads[0] // 2)
                    fts = nn.functional.pad(fts, pads_lr, mode=self.pad)
                    fts = nn.functional.pad(fts, pads_tb, mode="constant")
                    fts += temp[int(2 * self.num_steps - count - 1)]
                    count += 1
        return fts
