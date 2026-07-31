"""
HEALPix UNet: single forward-pass encoder–decoder stack.

Adapted from the modulus-uw ``physicsnemo.models.dlwp_healpix.HEALPixUNet``.
"""

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn

from .healpix_blocks import HEALPixBuildContext
from .healpix_decoder import UNetDecoderConfig
from .healpix_encoder import UNetEncoderConfig
from .healpix_layers import HEALPixFoldFaces, HEALPixUnfoldFaces


class HEALPixUNet(nn.Module):
    """Feed-forward UNet on the HEALPix mesh.

    The model operates on tensors with shape ``[B, F, C, H, W]`` where ``F=12``
    is the number of HEALPix faces and ``C`` is the channel count. Faces are
    folded into the batch dimension before the encoder/decoder and unfolded
    after the decoder, so the encoder/decoder operate on plain 4D tensors of
    shape ``[B*F, C, H, W]``.
    """

    CHANNEL_DIM = 2  # [B, F, C, H, W]

    def __init__(
        self,
        encoder: UNetEncoderConfig,
        decoder: UNetDecoderConfig,
        input_channels: int,
        output_channels: int,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: Sequence[int] | None = None,
    ):
        """
        Initialize the HEALPixUNet model.

        Args:
            encoder: Configuration for the U-net encoder.
            decoder: Configuration for the U-net decoder.
            input_channels: Number of channels in the input tensor (i.e. the
                size of the channel dimension of the tensor passed to
                ``forward``).
            output_channels: Number of channels in the output tensor.
            hpx_padding_mode: HEALPix padding backend. One of ``"earth2grid"``,
                ``"karlbauer"``, or ``"isolatitude"``. Default ``"earth2grid"``.
            nside: Face height/width per UNet level, shallowest to deepest.
                Length must equal ``len(encoder.n_channels)``. Required when
                ``hpx_padding_mode`` is ``"isolatitude"``. Child modules validate
                face size at runtime (e.g. ``HEALPixPaddingIsolatitude``).
        """
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hpx_padding_mode = hpx_padding_mode

        levels = len(encoder.n_channels)
        if len(decoder.n_channels) != levels:
            raise ValueError(
                "encoder and decoder must have same number of levels; got "
                f"{levels} vs {len(decoder.n_channels)}"
            )
        if hpx_padding_mode == "isolatitude" and nside is None:
            raise ValueError(
                'hpx_padding_mode="isolatitude" requires nside (one int per UNet level)'
            )
        nside_resolved: tuple[int, ...] | None
        if nside is not None:
            nside_levels = tuple(int(v) for v in nside)
            if len(nside_levels) != levels:
                raise ValueError(
                    f"nside length must match UNet levels; got {len(nside_levels)} "
                    f"vs {levels}"
                )
            if any(v < 1 for v in nside_levels):
                raise ValueError(f"nside values must be positive; got {nside_levels}")
            nside_resolved = nside_levels
        else:
            nside_resolved = None
        self.nside = nside_resolved

        build_ctx = HEALPixBuildContext(
            hpx_padding_mode=hpx_padding_mode,
            nside_levels=nside_resolved,
        )

        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces(num_faces=12)

        self.encoder = encoder.build(input_channels=input_channels, ctx=build_ctx)
        self.decoder = decoder.build(output_channels=output_channels, ctx=build_ctx)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Tensor of shape ``[B, F=12, input_channels, H, W]``.

        Returns:
            Tensor of shape ``[B, F=12, output_channels, H, W]``.
        """
        if inputs.ndim != 5:
            raise ValueError(
                "HEALPixUNet expects a 5D input [B, F, C, H, W]; got tensor "
                f"with shape {tuple(inputs.shape)}"
            )
        if inputs.shape[self.CHANNEL_DIM] != self.input_channels:
            raise ValueError(
                f"Expected input to have {self.input_channels} channels at "
                f"dim {self.CHANNEL_DIM}, got {inputs.shape[self.CHANNEL_DIM]}."
            )
        if self.nside is not None:
            h, w = inputs.shape[-2], inputs.shape[-1]
            expected = self.nside[0]
            if h != expected or w != expected:
                raise ValueError(
                    f"Input face size ({h}, {w}) does not match nside[0]={expected}"
                )

        folded = self.fold(inputs)
        encodings = self.encoder(folded)
        decodings = self.decoder(encodings)
        return self.unfold(decodings)
