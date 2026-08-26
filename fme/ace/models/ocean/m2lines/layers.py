from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.utils.checkpoint

from fme.core.models.conditional_sfno.layers import Context, ContextConfig

from .activations import CappedGELU


class BilinearUpsample(torch.nn.Module):
    def __init__(self, upsampling: int = 2, **kwargs):
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=upsampling, mode="bilinear")

    def forward(self, x):
        return self.upsampler(x)


class ZonallyPeriodicBilinearUpsample(torch.nn.Module):
    """Bilinear upsampling that enforces periodicity along the longitude axis.

    Adapted from the ``ZonallyPeriodicBilinearUpsample`` in m2lines/Samudra
    (https://github.com/m2lines/Samudra/blob/ab554631973ced3c567c1ef65ef2f84c222458d7/src/samudra/models/modules/blocks.py)
    A plain bilinear ``Upsample`` interpolates the longitude (width) boundary
    against a replicated edge column, which leaves a discontinuity at the lon=0
    seam. Here we pad one column on each longitude edge with the wrapped
    (circular) neighbor before interpolating, then crop the upsampled padding
    back off, so the seam is interpolated against its true periodic neighbor.
    The latitude (height) axis is left unpadded, consistent with the constant
    padding used for the latitude axis elsewhere in Samudra. The output shape
    matches ``BilinearUpsample``.
    """

    def __init__(self, upsampling: int = 2, **kwargs):
        super().__init__()
        self.upsampling = upsampling

    def forward(self, x):
        width = x.shape[-1]
        padded = torch.nn.functional.pad(x, (1, 1, 0, 0), mode="circular")
        upsampled = torch.nn.functional.interpolate(
            padded,
            scale_factor=self.upsampling,
            mode="bilinear",
            align_corners=False,
        )
        start = self.upsampling
        end = start + width * self.upsampling
        return upsampled[..., start:end]


class AvgPool(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(pooling)

    def forward(self, x):
        return self.avgpool(x)


class NoiseConditioning(torch.nn.Module):
    """Zero-initialized conditional scale and bias driven by a noise field.

    Applied immediately after a non-affine normalization layer, this turns that
    norm into a conditional one, the same construction ``ConditionalLayerNorm``
    uses for the SFNO: ``x -> x * (1 + W_scale(noise)) + W_bias(noise)``. Both
    convolutions are zero-initialized, so an untrained model is exactly
    deterministic and stochasticity is learned from the ensemble loss.

    The noise field is drawn at the model's input resolution, while a
    conditioned block may run at a coarser resolution inside the U-Net, so the
    noise is subsampled onto the block's grid before projection. Subsampling
    rather than area-averaging is what keeps the noise at unit variance at every
    injection depth: averaging iid noise over a k-by-k window divides its
    standard deviation by k, which on a 45x90 grid would hand the bottleneck
    block noise of std 0.05 and spread the amplitude 20-fold across the depths
    that ``all_blocks`` conditions. The SFNO reference has no such problem
    because its latent never changes resolution, so ``ConditionalLayerNorm``
    always sees std-1 noise; DLESyM-Ocean likewise draws its bottleneck noise
    at bottleneck resolution. Subsampling iid noise leaves it iid, so the
    coarse field is white noise of unit variance whatever the shape ratio is.
    """

    def __init__(self, n_channels: int, embed_dim_noise: int):
        super().__init__()
        self.W_scale = torch.nn.Conv2d(
            embed_dim_noise, n_channels, kernel_size=1, bias=False
        )
        self.W_bias = torch.nn.Conv2d(
            embed_dim_noise, n_channels, kernel_size=1, bias=False
        )
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        target = (x.shape[-2], x.shape[-1])
        if (noise.shape[-2], noise.shape[-1]) != target:
            if noise.shape[-2] < target[0] or noise.shape[-1] < target[1]:
                # subsampling only stays iid while it discards cells: upsampling
                # would duplicate them, correlating neighbours and breaking the
                # unit variance this resampling exists to preserve. Samudra
                # never asks for it (the noise is drawn at input resolution and
                # every block is coarser), so this is an invariant, not a case
                # to handle.
                raise ValueError(
                    f"noise field {tuple(noise.shape[-2:])} is coarser than the "
                    f"block grid {target}; it must be at least as fine."
                )
            noise = torch.nn.functional.interpolate(noise, size=target, mode="nearest")
        return x * (1.0 + self.W_scale(noise)) + self.W_bias(noise)


class ConvNeXtBlock(torch.nn.Module):
    """
    A convolution block as reported in https://github.com/CognitiveModeling/dlwp-hpx/blob/main/src/dlwp-hpx/dlwp/model/modules/blocks.py.
    This is a modified version of the actual ConvNextblock which
    is used in the HealPix paper.

    When ``context_config`` is given with a non-zero noise embedding, each
    normalization layer is followed by a ``NoiseConditioning`` scale and bias
    read off the ``Context`` passed to ``forward``, making the block's norms
    conditional, so it requires a normalization layer to condition (``norm`` not
    None). Samudra's default ``instance`` norm is built with ``affine=False``, so
    it is a pure normalizer and the conditional scale/bias is exactly a
    conditional instance norm.
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
        context_config: ContextConfig | None = None,
    ):
        super().__init__()
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"

        self.N_in = in_channels
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = pad
        self.norm = norm
        self.norm_kwargs: Mapping[str, Any] = {} if norm_kwargs is None else norm_kwargs
        self.checkpoint_strategy = checkpoint_strategy
        assert n_layers == 1, "Can only use a single layer here!"  # Needs fixing

        if context_config is None:
            embed_dim_noise = 0
        else:
            embed_dim_noise = context_config.embed_dim_noise
        if embed_dim_noise > 0 and norm is None:
            raise ValueError(
                "Noise conditioning requires a normalization layer to condition, "
                "but norm is None."
            )

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

        hidden_channels = int(in_channels * upscale_factor)

        # Convolution block. Layer order is unchanged from the unconditioned
        # block, so checkpoints trained before conditioning existed still load;
        # the conditioning modules live in a separate ModuleDict which is empty
        # (and so contributes no state) when conditioning is off.
        convblock: list[torch.nn.Module] = []
        norm_indices: list[int] = []
        convblock.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        norm_layers = self._build_norm(norm, hidden_channels)
        if norm_layers:
            norm_indices.append(len(convblock))
        convblock.extend(norm_layers)
        convblock.append(activation())

        convblock.append(
            torch.nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        norm_layers = self._build_norm(norm, hidden_channels)
        if norm_layers:
            norm_indices.append(len(convblock))
        convblock.extend(norm_layers)
        convblock.append(activation())

        # Linear postprocessing
        convblock.append(
            torch.nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )
        )
        self.convblock = torch.nn.ModuleList(convblock)

        self.noise_conditioning = torch.nn.ModuleDict()
        if embed_dim_noise > 0:
            for index in norm_indices:
                self.noise_conditioning[str(index)] = NoiseConditioning(
                    hidden_channels, embed_dim_noise
                )

    def _build_norm(self, norm: str | None, num_features: int) -> list[torch.nn.Module]:
        if norm == "batch":
            return [torch.nn.BatchNorm2d(num_features, **self.norm_kwargs)]
        elif norm == "instance":
            return [torch.nn.InstanceNorm2d(num_features, **self.norm_kwargs)]
        elif norm == "layer":
            return [torch.nn.LayerNorm(num_features, **self.norm_kwargs)]
        elif norm is None:
            return []
        raise NotImplementedError(f"Normalization {norm} not implemented")

    def _apply_simple_checkpoint(self, layer, x):
        if self.checkpoint_strategy == "simple" and not isinstance(layer, nn.Conv2d):
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            x = layer(x)
        return x

    def forward(self, x, context: Context | None = None):
        if len(self.noise_conditioning) > 0 and (
            context is None or context.noise is None
        ):
            raise ValueError(
                "This ConvNeXtBlock is noise-conditioned, so forward requires a "
                "Context carrying a noise field."
            )
        skip = self.skip_module(x)
        for i, layer in enumerate(self.convblock):
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
            if str(i) in self.noise_conditioning:
                assert context is not None and context.noise is not None
                x = self.noise_conditioning[str(i)](x, context.noise)
        return skip + x
