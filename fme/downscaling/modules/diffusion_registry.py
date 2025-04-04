import dataclasses
from typing import Any, List, Mapping, Optional, Protocol, Tuple, Type

import dacite
import torch

from fme.downscaling.modules.preconditioners import EDMPrecond
from fme.downscaling.modules.unet_diffusion import UNetDiffusionModule
from fme.downscaling.modules.unets import SongUNet


# TODO: Look into why we need to take in coarse and not target shape
class ModuleConfig(Protocol):
    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
    ) -> torch.nn.Module: ...


@dataclasses.dataclass
class PreBuiltBuilder:
    module: torch.nn.Module

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
    ) -> torch.nn.Module:
        return self.module


@dataclasses.dataclass
class UNetDiffusionSong:
    model_channels: int = 128
    channel_mult: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 2, 2])

    channel_mult_emb: int = 4
    num_blocks: int = 4
    attn_resolutions: List[int] = dataclasses.field(default_factory=lambda: [16])
    dropout: float = 0.10
    label_dropout: int = 0

    embedding_type: str = "positional"
    channel_mult_noise: int = 1
    encoder_type: str = "standard"
    decoder_type: str = "standard"
    resample_filter: List[int] = dataclasses.field(default_factory=lambda: [1, 1])

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
    ):
        target_height, target_width = [s * downscale_factor for s in coarse_shape]
        # number of input channels = latents (num desired outputs) + conditioning fields
        n_in_channels_conditioned = n_in_channels + n_out_channels
        unet = SongUNet(
            min(target_height, target_width),
            n_in_channels_conditioned,
            n_out_channels,
            model_channels=self.model_channels,
            channel_mult=self.channel_mult,
            channel_mult_emb=self.channel_mult_emb,
            num_blocks=self.num_blocks,
            attn_resolutions=self.attn_resolutions,
            dropout=self.dropout,
            label_dropout=self.label_dropout,
            embedding_type=self.embedding_type,
            channel_mult_noise=self.channel_mult_noise,
            encoder_type=self.encoder_type,
            decoder_type=self.decoder_type,
            resample_filter=self.resample_filter,
        )
        return UNetDiffusionModule(
            EDMPrecond(
                unet,
                sigma_data=sigma_data,
            ),
        )


@dataclasses.dataclass
class DiffusionModuleRegistrySelector:
    """
    Model architecture selector for diffusion models.

    Parameters:
        type: key of the model architecture to use
        config: configuration for the model architecture
        expects_interpolated_input: whether the model expects interpolated input
            to the target resolution
    """

    type: str
    config: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    expects_interpolated_input: Optional[bool] = None

    def __post_init__(self):
        if self.type == "prebuilt" and self.expects_interpolated_input is None:
            raise ValueError(
                "If using a prebuilt module, you must specify whether it expects "
                "interpolated input."
            )
        if self.expects_interpolated_input is None:
            self.expects_interpolated_input = EXPECTS_INTERPOLATED.get(self.type, False)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
    ) -> torch.nn.Module:
        return dacite.from_dict(
            data_class=NET_REGISTRY[self.type],
            data=self.config,
            config=dacite.Config(strict=True),
        ).build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            sigma_data=sigma_data,
        )


NET_REGISTRY: Mapping[str, Type[ModuleConfig]] = {
    "unet_diffusion_song": UNetDiffusionSong,
    "prebuilt": PreBuiltBuilder,
}


EXPECTS_INTERPOLATED = {
    "unet_diffusion_song": True,
}
