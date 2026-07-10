import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

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
    n_vector_outputs : int, optional
        Number of output channels produced by an MLP readout head instead of the
        final convolution, by default 0 (disabled). When > 0, the last
        ``n_vector_outputs`` of the ``output_channels`` are predicted as a
        per-sample vector by globally average-pooling the penultimate feature map
        and passing it through an MLP, then broadcasting each scalar across the
        spatial grid. This guarantees those outputs are spatially homogeneous by
        construction (useful for scalar targets such as a Nino3.4 index vector).
        The remaining ``output_channels - n_vector_outputs`` are produced by the
        usual final convolution and are ordered first.
    vector_hidden_dim : int, optional
        Hidden dimension of the readout MLP; defaults to the penultimate feature
        width. Ignored when ``n_vector_outputs == 0``.

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
        n_vector_outputs: int = 0,
        vector_hidden_dim: int | None = None,
    ):
        super().__init__()

        if n_vector_outputs < 0:
            raise ValueError("n_vector_outputs must be non-negative")
        if n_vector_outputs >= output_channels:
            raise ValueError(
                "n_vector_outputs must be smaller than output_channels "
                f"(got n_vector_outputs={n_vector_outputs}, "
                f"output_channels={output_channels})"
            )

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_vector_outputs = n_vector_outputs
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
                    upscale_factor=self.upscale_factor,
                    checkpoint_strategy=self.checkpoint_strategy,
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
                upscale_factor=self.upscale_factor,
                checkpoint_strategy=self.checkpoint_strategy,
            )
        )
        # The final convolution produces only the spatial (field) outputs; any
        # vector-readout outputs are produced by the MLP head below and appended.
        n_field_outputs = self.output_channels - self.n_vector_outputs
        layers.append(torch.nn.Conv2d(b, n_field_outputs, self.last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width_with_input) - 1)

        # MLP readout head: globally pool the penultimate feature map (``b``
        # channels feed the final convolution) and map to a per-sample vector.
        if self.n_vector_outputs > 0:
            hidden_dim = vector_hidden_dim if vector_hidden_dim is not None else b
            self.vector_readout: nn.Module = nn.Sequential(
                nn.Linear(b, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.n_vector_outputs),
            )

    def forward(self, fts):
        temp: list[torch.Tensor] = []
        count = 0
        penultimate: torch.Tensor | None = None
        for layer in self.layers:
            crop = fts.shape[2:]
            if isinstance(layer, nn.Conv2d):
                # Capture the feature map feeding the final convolution; the MLP
                # readout head reads it (before padding) to produce the vector.
                penultimate = fts
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            if self.checkpoint_strategy == "all":
                fts = torch.utils.checkpoint.checkpoint(layer, fts, use_reentrant=False)
            else:
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
                    pads_lr = (pads[1] // 2, pads[1] - pads[1] // 2, 0, 0)
                    pads_tb = (0, 0, pads[0] // 2, pads[0] - pads[0] // 2)
                    fts = nn.functional.pad(fts, pads_lr, mode=self.pad)
                    fts = nn.functional.pad(fts, pads_tb, mode="constant")
                    fts += temp[int(2 * self.num_steps - count - 1)]
                    count += 1
        if self.n_vector_outputs > 0:
            assert penultimate is not None  # a final Conv2d always runs
            # Global average pool over the spatial dims -> (batch, channels).
            pooled = penultimate.mean(dim=(-2, -1))
            vector = self.vector_readout(pooled)  # (batch, n_vector_outputs)
            # Broadcast each scalar across the spatial grid so the output stays
            # a standard (batch, channel, lat, lon) tensor; these channels are
            # spatially homogeneous by construction.
            height, width = fts.shape[-2], fts.shape[-1]
            vector_field = vector[..., None, None].expand(-1, -1, height, width)
            fts = torch.cat([fts, vector_field], dim=1)
        return fts
