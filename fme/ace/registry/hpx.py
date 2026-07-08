import dataclasses
from collections.abc import Sequence
from typing import Literal

import torch.nn as nn

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
        enable_nhwc: Use NHWC tensor layout for child modules.
        hpx_padding_mode: HEALPix padding backend (``"earth2grid"``,
            ``"karlbauer"``, ``"isolatitude"``). Default ``"earth2grid"``.
        nside: Face height/width per UNet level (shallowest to deepest). Required for
            ``isolatitude`` padding.
    """

    encoder: UNetEncoderConfig
    decoder: UNetDecoderConfig
    enable_nhwc: bool = False
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
        return HEALPixUNet(
            encoder=self.encoder,
            decoder=self.decoder,
            input_channels=n_in_channels,
            output_channels=n_out_channels,
            enable_nhwc=self.enable_nhwc,
            hpx_padding_mode=self.hpx_padding_mode,
            nside=self.nside,
        )
