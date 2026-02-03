import dataclasses
from collections.abc import Mapping
from typing import Any, Literal, Protocol

import dacite
import torch

from fme.downscaling.modules.unet_diffusion import UNetDiffusionModule
from fme.downscaling.modules.vendorized import EDMPrecond, SongUNet, SongUNetv2


# TODO: Look into why we need to take in coarse and not target shape
class ModuleConfig(Protocol):
    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
        use_channels_last: bool = True,
        use_amp_bf16: bool = False,
    ) -> torch.nn.Module: ...


@dataclasses.dataclass
class PreBuiltBuilder:
    module: torch.nn.Module

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
        use_channels_last: bool = False,
        use_amp_bf16: bool = False,
    ) -> torch.nn.Module:
        return self.module


@dataclasses.dataclass
class UNetDiffusionSong:
    model_channels: int = 128
    channel_mult: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 2, 2])

    channel_mult_emb: int = 4
    num_blocks: int = 4
    attn_resolutions: list[int] = dataclasses.field(default_factory=lambda: [16])
    dropout: float = 0.10
    label_dropout: int = 0

    embedding_type: str = "positional"
    channel_mult_noise: int = 1
    encoder_type: str = "standard"
    decoder_type: str = "standard"
    resample_filter: list[int] = dataclasses.field(default_factory=lambda: [1, 1])

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
        use_channels_last: bool = False,
        use_amp_bf16: bool = False,
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
            use_amp_bf16=use_amp_bf16,
            use_channels_last=False,
        )


@dataclasses.dataclass
class UNetDiffusionSongv2:
    model_channels: int = 128
    channel_mult: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 2, 2])

    channel_mult_emb: int = 4
    num_blocks: int = 4
    attn_resolutions: list[int] = dataclasses.field(default_factory=lambda: [16])
    dropout: float = 0.10
    label_dropout: int = 0

    embedding_type: Literal["fourier", "positional", "zero"] = "positional"
    channel_mult_noise: int = 1
    encoder_type: Literal["standard", "skip", "residual"] = "standard"
    decoder_type: Literal["standard", "skip"] = "standard"
    resample_filter: list[int] = dataclasses.field(default_factory=lambda: [1, 1])
    act: str = "silu"
    use_apex_gn: bool = True

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: tuple[int, int],
        downscale_factor: int,
        sigma_data: float,
        use_channels_last: bool = True,
        use_amp_bf16: bool = True,
    ):
        target_height, target_width = [s * downscale_factor for s in coarse_shape]
        # number of input channels = latents (num desired outputs) + conditioning fields
        n_in_channels_conditioned = n_in_channels + n_out_channels
        unet = SongUNetv2(
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
            act=self.act,
            use_apex_gn=self.use_apex_gn,
            amp_mode=use_amp_bf16,
        )
        module = UNetDiffusionModule(
            EDMPrecond(
                unet,
                sigma_data=sigma_data,
            ),
            use_channels_last=use_channels_last,
            use_amp_bf16=use_amp_bf16,
        )
        if use_channels_last:
            module = module.to(memory_format=torch.channels_last)
        return module


@dataclasses.dataclass
class DiffusionModuleRegistrySelector:
    """
    Model architecture selector for diffusion models.

    Parameters:
        type: key of the model architecture to use
        config: configuration for the model architecture
        expects_interpolated_input: whether the model expects interpolated input
            to the target resolution
        use_channels_last: whether to use channels_last memory format for the model
            and forward pass inputs. This can provide a speedup on modern
            NVIDIA GPUs (H100, B200) by optimizing memory layout for Tensor Cores.
            Only supported for SongUNetv2. Other models will ignore this flag.
        use_amp_bf16: whether to use automatic mixed precision with bfloat16
            during forward pass.
    """

    type: str
    config: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    expects_interpolated_input: bool | None = None
    use_channels_last: bool = False
    use_amp_bf16: bool = False

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
        coarse_shape: tuple[int, int],
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
            use_channels_last=self.use_channels_last,
            use_amp_bf16=self.use_amp_bf16,
        )


NET_REGISTRY: Mapping[str, type[ModuleConfig]] = {
    "unet_diffusion_song": UNetDiffusionSong,
    "unet_diffusion_song_v2": UNetDiffusionSongv2,
    "prebuilt": PreBuiltBuilder,
}


EXPECTS_INTERPOLATED = {
    "unet_diffusion_song": True,
    "unet_diffusion_song_v2": True,
}
