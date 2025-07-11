import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fme.ace.models.ocean.m2lines.layers import AvgPool, BilinearUpsample, ConvNeXtBlock
from fme.ace.models.ocean.m2lines.utils import pairwise


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
            )
        )
        layers.append(BilinearUpsample(in_channels=b, out_channels=b))
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
                )
            )
            layers.append(BilinearUpsample(in_channels=b, out_channels=b))
        layers.append(
            ConvNeXtBlock(
                b,
                b,
                dilation=dilation_reversed[i],
                n_layers=n_layers_reversed[i],
                pad=self.pad,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
            )
        )
        layers.append(torch.nn.Conv2d(b, self.output_channels, self.last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width_with_input) - 1)

    def forward(self, fts):
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
            fts = layer(fts)
            if count < self.num_steps:
                if isinstance(layer, ConvNeXtBlock):
                    temp.append(fts)
                    count += 1
            elif count >= self.num_steps:
                if isinstance(layer, BilinearUpsample):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(
                        temp[int(2 * self.num_steps - count - 1)].shape[2:]
                    )
                    pads = shape - crop
                    pads = [
                        pads[1] // 2,
                        pads[1] - pads[1] // 2,
                        pads[0] // 2,
                        pads[0] - pads[0] // 2,
                    ]
                    fts = nn.functional.pad(fts, pads)
                    fts += temp[int(2 * self.num_steps - count - 1)]
                    count += 1
        return fts
