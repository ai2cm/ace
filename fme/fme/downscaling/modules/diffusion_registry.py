import dataclasses
from typing import Any, List, Mapping, Optional, Tuple, Type

import dacite
import torch

from fme.downscaling.modules.preconditioners import EDMPrecond
from fme.downscaling.modules.registry import ModuleConfig, compute_unet_padding_size
from fme.downscaling.modules.unet_diffusion import UNetDiffusionModule
from fme.downscaling.modules.unets import SongUNet


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

    # diffusion parameters
    sigma_min: float = 0.0
    sigma_max: float = float("inf")
    sigma_data: float = 0.5

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        fine_topography: Optional[torch.Tensor],
    ):
        divisor = 2 ** (len(self.channel_mult) - 1)
        target_height, target_width = [
            s * downscale_factor
            + compute_unet_padding_size(s * downscale_factor, divisor)
            for s in coarse_shape
        ]
        n_in_channels_conditioned = 2 * n_in_channels
        unet = SongUNet(
            min(target_height, target_width),
            (
                n_in_channels_conditioned
                if fine_topography is None
                else n_in_channels_conditioned + 1
            ),
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
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                sigma_data=self.sigma_data,
            ),
            coarse_shape,
            (target_height, target_width),
            downscale_factor,
            fine_topography,
        )


@dataclasses.dataclass
class DiffusionModuleRegistrySelector:
    type: str
    config: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        fine_topography: Optional[torch.Tensor],
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
            fine_topography=fine_topography,
        )


NET_REGISTRY: Mapping[str, Type[ModuleConfig]] = {
    "unet_diffusion_song": UNetDiffusionSong
}
