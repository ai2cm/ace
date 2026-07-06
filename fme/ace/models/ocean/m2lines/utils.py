from itertools import tee
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class MaskedAvgPool2d(nn.Module):
    def __init__(
        self,
        pooling: int = 2,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        self.pooling = pooling
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        numerator = F.avg_pool2d(
            x * mask,
            kernel_size=self.pooling,
            stride=self.pooling,
        )

        valid_fraction = F.avg_pool2d(
            mask,
            kernel_size=self.pooling,
            stride=self.pooling,
        )

        output = numerator / valid_fraction.clamp_min(self.eps)
        output_mask = (valid_fraction > 0).to(dtype=x.dtype)

        return output * output_mask, output_mask

class PartialDepthwiseConv2d(nn.Module):
    """
    Partial spatial convolution applied independently to each channel.

    x:    [B, C, H, W]
    mask: [B, C, H, W], with 1 for valid and 0 for invalid

    Longitude is padded periodically; latitude is padded as invalid.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
        longitude_padding: str = "circular",
        eps: float = 1.0e-6,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1) // 2
        self.longitude_padding = longitude_padding
        self.eps = eps

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=bias,
        )

        # One mask-counting kernel per channel.
        self.register_buffer(
            "mask_kernel",
            torch.ones(channels, 1, kernel_size, kernel_size),
            persistent=False,
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        p = self.padding

        x = F.pad(
            x,
            (p, p, 0, 0),
            mode=self.longitude_padding,
        )

        # Outside the latitude boundary is invalid.
        x = F.pad(
            x,
            (0, 0, p, p),
            mode="constant",
            value=0.0,
        )

        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        if mask.shape[0] == 1 and x.shape[0] > 1:
            mask = mask.expand(x.shape[0], -1, -1, -1)

        if mask.shape[1] == 1:
            mask = mask.expand(-1, self.channels, -1, -1)

        if mask.shape != x.shape:
            raise ValueError(
                f"x has shape {x.shape}, but mask has shape {mask.shape}"
            )

        mask = mask.to(dtype=x.dtype)

        masked_x = self._pad(x * mask)
        padded_mask = self._pad(mask)

        raw_output = self.conv(masked_x)

        valid_count = F.conv2d(
            padded_mask,
            self.mask_kernel.to(dtype=x.dtype),
            bias=None,
            stride=self.conv.stride,
            dilation=self.conv.dilation,
            groups=self.channels,
        )

        full_count = float(self.kernel_size * self.kernel_size)
        scale = full_count / valid_count.clamp_min(1.0)

        if self.conv.bias is not None:
            bias = self.conv.bias.view(1, -1, 1, 1)
            output = (raw_output - bias) * scale + bias
        else:
            output = raw_output * scale

        output_mask = (valid_count > 0).to(dtype=x.dtype)
        output = output * output_mask

        return output, output_mask