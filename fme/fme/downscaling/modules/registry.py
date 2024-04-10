"""Registry for downscaling modules. Note that all modules should accept and
return tensors of shape (batch, channel, height, width)."""

import dataclasses
from typing import Any, Literal, Mapping, Protocol, Sequence, Tuple, Type

import dacite
import torch

from fme.downscaling.modules.swinir import SwinIR


class ModuleConfig(Protocol):
    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
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
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
    ):
        height, width = coarse_shape
        # TODO(gideond): The SwinIR docs appear to be wrong, dig into why these
        # need to take these values to give the right output shapes
        height = (height // downscale_factor // self.window_size + 1) * self.window_size
        width = (width // downscale_factor // self.window_size + 1) * self.window_size
        return SwinIR(
            upscale=downscale_factor,  # different ML versus climate convention
            img_size=(height, width),  # type: ignore
            window_size=self.window_size,
            depths=self.depths,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            upsampler=self.upsampler,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
        )


@dataclasses.dataclass
class ModuleRegistrySelector:
    type: str
    config: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
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
        )


@dataclasses.dataclass
class PreBuiltBuilder:
    module: torch.nn.Module

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
    ) -> torch.nn.Module:
        return self.module


class Interpolate(torch.nn.Module):
    def __init__(self, downscale_factor: int, mode: Literal["bicubic", "nearest"]):
        super(Interpolate, self).__init__()
        self.downscale_factor = downscale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor):
        if self.mode == "bicubic":
            align_corners = True
        else:
            align_corners = None
        return torch.nn.functional.interpolate(
            x,
            scale_factor=[self.downscale_factor, self.downscale_factor],
            mode=self.mode,
            align_corners=align_corners,
        )


@dataclasses.dataclass
class InterpolateConfig:
    mode: Literal["bicubic", "nearest"]

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
    ) -> torch.nn.Module:
        return Interpolate(downscale_factor, self.mode)


NET_REGISTRY: Mapping[str, Type[ModuleConfig]] = {
    "swinir": SwinirConfig,
    "prebuilt": PreBuiltBuilder,
    "interpolate": InterpolateConfig,
}
