import dataclasses
from typing import Optional, Sequence

import torch.nn as nn

from fme.ace.models.healpix.healpix_decoder import UNetDecoderConfig
from fme.ace.models.healpix.healpix_encoder import UNetEncoderConfig
from fme.ace.models.healpix.healpix_recunet import HEALPixRecUNet
from fme.ace.models.healpix.healpix_unet import HEALPixUNet
from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.dataset_info import DatasetInfo


@ModuleSelector.register("HEALPixRecUNet")
@dataclasses.dataclass
class HEALPixRecUNetBuilder(ModuleConfig):
    """
    Configuration for the HEALPixRecUNet architecture used in DLWP.

    Parameters:
        presteps: Number of pre-steps, by default 1.
        input_time_size: Input time dimension, by default 0.
        output_time_size: Output time dimension, by default 0.
        delta_time: Delta time interval, by default "6h".
        reset_cycle: Reset cycle interval, by default "24h".
        input_channels: Number of input channels, by default 8.
        output_channels: Number of output channels, by default 8.
        n_constants: Number of constant input channels, by default 2.
        decoder_input_channels: Number of input channels for the decoder, by default 1.
        enable_nhwc: Flag to enable NHWC data format, by default False.
        enable_healpixpad: Deprecated legacy padding toggle; omit (``None``) when using
            ``hpx_padding_mode``. If explicitly ``True``/``False`` with no ``hpx_padding_mode``,
            legacy mapping still applies with a deprecation warning.
    """

    encoder: UNetEncoderConfig
    decoder: UNetDecoderConfig
    presteps: int = 1
    input_time_size: int = 0
    output_time_size: int = 0
    delta_time: str = "6h"
    reset_cycle: str = "24h"
    n_constants: int = 2
    decoder_input_channels: int = 1
    prognostic_variables: int = 7
    enable_nhwc: bool = False
    enable_healpixpad: Optional[bool] = None
    hpx_padding_mode: Optional[str] = None
    compile_padding: bool = False
    nside: Optional[Sequence[int] | int] = None

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        """
        Builds the HEALPixRecUNet model.

        Args:
            n_in_channels: Number of input channels.
            n_out_channels: Number of output channels.
            dataset_info: Information about the dataset.

        Returns:
            HEALPixRecUNet model.
        """
        if len(dataset_info.all_labels) > 0:
            raise ValueError("HEALPixRecUNet does not support labels")
        # Construct the HEALPixRecUNet module here using the parameters
        return HEALPixRecUNet(
            encoder=self.encoder,
            decoder=self.decoder,
            input_channels=n_in_channels,
            output_channels=n_out_channels,
            prognostic_variables=self.prognostic_variables,
            n_constants=self.n_constants,
            decoder_input_channels=self.decoder_input_channels,
            input_time_size=self.input_time_size,
            output_time_size=self.output_time_size,
            delta_time=self.delta_time,
            reset_cycle=self.reset_cycle,
            presteps=self.presteps,
            enable_nhwc=self.enable_nhwc,
            enable_healpixpad=self.enable_healpixpad,
            hpx_padding_mode=self.hpx_padding_mode,
            compile_padding=self.compile_padding,
            nside=self.nside,
        )


@ModuleSelector.register("HEALPixUNet")
@dataclasses.dataclass
class HEALPixUNetBuilder(ModuleConfig):
    """
    Configuration for the non-recurrent HEALPixUNet architecture.

    Unlike :class:`HEALPixRecUNetBuilder`, this builder produces a stateless
    UNet: time stepping, prognostic/diagnostic splitting, and residual
    prediction are all expected to be handled by the stepper.

    Parameters:
        encoder: UNet encoder configuration.
        decoder: UNet decoder configuration. ``recurrent_block`` must be
            ``None``.
        enable_nhwc: Use NHWC tensor layout for child modules.
        enable_healpixpad: Deprecated legacy padding toggle. Prefer
            ``hpx_padding_mode``.
        hpx_padding_mode: HEALPix padding backend (``"earth2grid"``,
            ``"karlbauer"``, ``"isolatitude"``).
        compile_padding: If ``True``, apply ``torch.compile`` to isolatitude
            padding modules.
        nside: Face size(s) per UNet level (shallowest to deepest).
    """

    encoder: UNetEncoderConfig
    decoder: UNetDecoderConfig
    enable_nhwc: bool = False
    enable_healpixpad: Optional[bool] = None
    hpx_padding_mode: Optional[str] = None
    compile_padding: bool = False
    nside: Optional[Sequence[int] | int] = None

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
            enable_healpixpad=self.enable_healpixpad,
            hpx_padding_mode=self.hpx_padding_mode,
            compile_padding=self.compile_padding,
            nside=self.nside,
        )
