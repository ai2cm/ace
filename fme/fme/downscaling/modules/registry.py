"""Registry for downscaling modules. Note that all modules should accept and
return tensors of shape (batch, channel, height, width)."""

import dataclasses
from typing import Any, Mapping, Protocol, Sequence, Tuple, Type

import dacite
import torch

from fme.downscaling.modules.swinir import SwinIR


class ModuleConfig(Protocol):
    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        lowres_shape: Tuple[int, int],
        upscale_factor: int,
    ) -> torch.nn.Module:
        ...


@dataclasses.dataclass
class SwinirConfig:
    window_size: int = 4
    depths: Sequence[int] = (6, 6, 6, 6)
    embed_dim: int = 60
    num_heads: Sequence[int] = (6, 6, 6, 6)
    mlp_ratio: int = 2
    upsampler: str = "pixelshuffledirect"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        lowres_shape: Tuple[int, int],
        upscale_factor: int,
    ):
        del n_out_channels  # unused for now
        height, width = lowres_shape
        # TODO(gideond): The SwinIR docs appear to be wrong, dig into why these
        # need to take these values to give the right output shapes
        height = (height // upscale_factor // self.window_size + 1) * self.window_size
        width = (width // upscale_factor // self.window_size + 1) * self.window_size
        return SwinIR(
            upscale=upscale_factor,
            img_size=(height, width),  # type: ignore
            window_size=self.window_size,
            depths=self.depths,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            upsampler=self.upsampler,
            in_chans=n_in_channels,
        )


@dataclasses.dataclass
class ModuleRegistrySelector(ModuleConfig):
    type: str
    config: Mapping[str, Any]

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        lowres_shape: Tuple[int, int],
        upscale_factor: int,
    ) -> torch.nn.Module:
        return dacite.from_dict(
            data_class=NET_REGISTRY[self.type],
            data=self.config,
            config=dacite.Config(strict=True),
        ).build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            lowres_shape=lowres_shape,
            upscale_factor=upscale_factor,
        )


@dataclasses.dataclass
class PreBuiltBuilder:
    module: torch.nn.Module

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        lowres_shape: Tuple[int, int],
        upscale_factor: int,
    ) -> torch.nn.Module:
        return self.module


NET_REGISTRY: Mapping[str, Type[ModuleConfig]] = {
    "swinir": SwinirConfig,
    "prebuilt": PreBuiltBuilder,
}
