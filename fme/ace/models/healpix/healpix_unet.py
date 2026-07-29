"""
HEALPix UNet: single forward-pass encoder–decoder stack.

Adapted from the modulus-uw ``physicsnemo.models.dlwp_healpix.HEALPixUNet``.
"""

import torch
import torch.nn as nn

from .healpix_layers import HEALPixFoldFaces, HEALPixUnfoldFaces


class HEALPixUNet(nn.Module):
    """Feed-forward UNet on the HEALPix mesh.

    The model operates on tensors with shape ``[B, F, C, H, W]`` where ``F=12``
    is the number of HEALPix faces and ``C`` is the channel count. Faces are
    folded into the batch dimension before the encoder/decoder and unfolded
    after the decoder, so the encoder/decoder operate on plain 4D tensors of
    shape ``[B*F, C, H, W]``.

    The encoder and decoder modules are built and validated by
    :class:`~fme.ace.registry.hpx.HEALPixUNetBuilder`; this module receives the
    built collaborators and the runtime scalars it needs and does no building
    itself.
    """

    CHANNEL_DIM = 2  # [B, F, C, H, W]

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        input_channels: int,
        output_channels: int,
        nside: tuple[int, ...] | None = None,
    ):
        """
        Initialize the HEALPixUNet model.

        Args:
            encoder: Built U-net encoder module.
            decoder: Built U-net decoder module.
            input_channels: Number of channels in the input tensor (i.e. the
                size of the channel dimension of the tensor passed to
                ``forward``).
            output_channels: Number of channels in the output tensor.
            nside: Resolved face height/width per UNet level, shallowest to
                deepest, or ``None``. ``forward`` checks the input face size
                against ``nside[0]`` when set.
        """
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.nside = nside

        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces(num_faces=12)

        self.encoder = encoder
        self.decoder = decoder

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
