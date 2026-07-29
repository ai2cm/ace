import dataclasses
from collections.abc import Sequence
from typing import Literal

import torch.nn as nn

from fme.ace.models.healpix.healpix_blocks import HEALPixBuildContext
from fme.ace.models.healpix.healpix_decoder import UNetDecoderConfig
from fme.ace.models.healpix.healpix_encoder import UNetEncoderConfig
from fme.ace.models.healpix.healpix_unet import HEALPixUNet
from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.dataset_info import DatasetInfo


@ModuleSelector.register("HEALPixUNet")
@dataclasses.dataclass
class HEALPixUNetBuilder(ModuleConfig):
    """
    Configuration for the HEALPix UNet (feed-forward encoder–decoder stack).

    Time stepping, multi-step inputs, residual prediction, and rollout live in
    the stepper, not in this module.

    Parameters:
        encoder: UNet encoder configuration.
        decoder: UNet decoder configuration.
        hpx_padding_mode: HEALPix padding backend applied to all child modules
            (``"earth2grid"``, ``"karlbauer"``, ``"isolatitude"``). Default
            ``"earth2grid"``.
        nside: Face height/width per UNet level (shallowest to deepest). Required for
            ``isolatitude`` padding.
    """

    encoder: UNetEncoderConfig
    decoder: UNetDecoderConfig
    hpx_padding_mode: Literal["earth2grid", "karlbauer", "isolatitude"] = "earth2grid"
    nside: Sequence[int] | None = None

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        """
        Build a HEALPixUNet model.

        Args:
            n_in_channels: Number of input channels.
            n_out_channels: Number of output channels.
            dataset_info: Information about the dataset.

        Returns:
            HEALPixUNet model.
        """
        if len(dataset_info.all_labels) > 0:
            raise ValueError("HEALPixUNet does not support labels")
        return self._build(
            input_channels=n_in_channels,
            output_channels=n_out_channels,
        )

    def _build(
        self,
        input_channels: int,
        output_channels: int,
    ) -> HEALPixUNet:
        """Build the encoder/decoder modules and assemble the HEALPixUNet.

        Constructs the shared ``HEALPixBuildContext``, validates the
        encoder/decoder level counts and the ``nside`` sequence, builds the
        encoder and decoder ``nn.Module``s, and passes the built collaborators
        plus the runtime scalars into :class:`HEALPixUNet`.
        """
        levels = len(self.encoder.n_channels)
        if len(self.decoder.n_channels) != levels:
            raise ValueError(
                "encoder and decoder must have same number of levels; got "
                f"{levels} vs {len(self.decoder.n_channels)}"
            )
        if self.hpx_padding_mode == "isolatitude" and self.nside is None:
            raise ValueError(
                'hpx_padding_mode="isolatitude" requires nside (one int per UNet level)'
            )
        nside_resolved: tuple[int, ...] | None
        if self.nside is not None:
            nside_levels = tuple(int(v) for v in self.nside)
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

        build_ctx = HEALPixBuildContext(
            hpx_padding_mode=self.hpx_padding_mode,
            nside_levels=nside_resolved,
        )
        encoder = self.encoder.build(input_channels=input_channels, ctx=build_ctx)
        decoder = self.decoder.build(output_channels=output_channels, ctx=build_ctx)
        return HEALPixUNet(
            encoder=encoder,
            decoder=decoder,
            input_channels=input_channels,
            output_channels=output_channels,
            nside=nside_resolved,
        )
