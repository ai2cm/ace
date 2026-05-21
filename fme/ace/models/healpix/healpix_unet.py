"""
HEALPix UNet: single forward-pass encoder–decoder stack.

Adapted from the modulus-uw ``physicsnemo.models.dlwp_healpix.HEALPixUNet``.
"""

from typing import Literal, Sequence

import torch as th
import torch.nn as nn

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
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal["earth2grid", "karlbauer", "isolatitude"] = "earth2grid",
        nside: Sequence[int] | int | None = (64, 32, 16),
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
            enable_nhwc: Use NHWC tensor layout for child modules.
            hpx_padding_mode: HEALPix padding backend. One of ``"earth2grid"``,
                ``"karlbauer"``, or ``"isolatitude"``. Default ``"earth2grid"``.
            nside: Face size(s) per UNet level (shallowest to deepest). May be
                a sequence with ``len(encoder.n_channels)`` entries, an int
                (treated as the shallowest level with halving per level), or
                ``None`` (defaults to ``64`` halving per level).
        """
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.enable_nhwc = enable_nhwc
        self.hpx_padding_mode = hpx_padding_mode

        levels = len(encoder.n_channels)
        if isinstance(nside, int):
            nside_levels = tuple(max(1, nside // (2**i)) for i in range(levels))
        elif nside is None:
            nside_levels = tuple(max(1, 64 // (2**i)) for i in range(levels))
        else:
            nside_levels = tuple(int(v) for v in nside)
        if len(nside_levels) != levels:
            raise ValueError(
                f"nside length must match UNet levels; got {len(nside_levels)} "
                f"vs {levels}"
            )
        if len(decoder.n_channels) != levels:
            raise ValueError(
                "encoder and decoder must have same number of levels for "
                "nside mapping"
            )
        self.nside = nside_levels

        self.fold = HEALPixFoldFaces(enable_nhwc=enable_nhwc)
        self.unfold = HEALPixUnfoldFaces(num_faces=12, enable_nhwc=enable_nhwc)

        encoder.input_channels = input_channels
        encoder.enable_nhwc = enable_nhwc
        encoder.hpx_padding_mode = self.hpx_padding_mode
        encoder.nside = self.nside[0]
        self.encoder = encoder.build()

        decoder.output_channels = output_channels
        decoder.enable_nhwc = enable_nhwc
        decoder.hpx_padding_mode = self.hpx_padding_mode
        decoder.nside = self.nside[-1]
        self.decoder = decoder.build()

    def forward(self, inputs: th.Tensor) -> th.Tensor:
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

        folded = self.fold(inputs)
        encodings = self.encoder(folded)
        decodings = self.decoder(encodings)
        return self.unfold(decodings)
