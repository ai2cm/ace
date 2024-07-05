"""Registry for downscaling modules. Note that all modules should accept and
return tensors of shape (batch, channel, height, width)."""

import dataclasses
from typing import (
    Any,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
)

import dacite
import torch

from fme.core.device import get_device
from fme.downscaling.metrics_and_maths import interpolate
from fme.downscaling.modules.swinir import SwinIR
from fme.downscaling.modules.unets import DhariwalUNet, SongUNet


class ModuleConfig(Protocol):
    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        fine_topography: Optional[torch.Tensor],
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
        fine_topography: Optional[torch.Tensor],
    ):
        if fine_topography is not None:
            raise NotImplementedError("Fine grid topography not supported by SwinIR")

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


def compute_unet_padding_size(value: int, divisor: int) -> int:
    """Compute the padding size so that `value + padding_size` is divisible by
    `divisor`."""
    remainder = value % divisor
    if remainder == 0:
        return 0
    return divisor - remainder


def pad_right(tensor: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """Pad the right side of the tensor with zeros to match the target shape."""
    return torch.nn.functional.pad(
        tensor,
        [
            0,
            target_shape[-1] - tensor.shape[-1],
            0,
            target_shape[-2] - tensor.shape[-2],
        ],
        mode="constant",
        value=0,
    )


class UNetRegressionModule(torch.nn.Module):
    """
    Performs downscaling by (1) bicubic interpolation to the fine grid (see [1]
    for where this is implemented in their source code and Fig. 1 (bottom left)
    of [2]), (2) padding with zeros to match the usual assumption that the
    shapes of the inputs contain many 2's in the prime factorization.

    [1] https://github.com/NVIDIA/modulus/blob/main/examples/generative/corrdiff/datasets/cwb.py#L491  # noqa: E501
    [2] https://arxiv.org/abs/2309.15214

    Args:
        unet: The U-Net model.
        coarse_shape: The shape of the coarse input.
        target_shape: The input and output shape required by the u-net.
        downscale_factor: The factor by which the coarse input is downscaled to
            the target output.
        fine_topography: Optional fine topography to condition the latent
            variables on.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        coarse_shape: Tuple[int, int],
        target_shape: Tuple[int, int],
        downscale_factor: int,
        fine_topography: Optional[torch.Tensor],
    ):
        super(UNetRegressionModule, self).__init__()
        self.unet = unet
        self.coarse_shape = coarse_shape
        self.target_shape = target_shape
        self.downscale_factor = downscale_factor
        self.fine_topography = fine_topography

    def forward(self, x: torch.Tensor):
        inputs = interpolate(x, self.downscale_factor)
        inputs = pad_right(inputs, self.target_shape)

        if self.fine_topography is not None:
            batch_size = inputs.shape[0]
            topography = (
                self.fine_topography.unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                .to(get_device())
            )
            topography = pad_right(topography, self.target_shape)
            inputs = torch.concat((inputs, topography), dim=1)

        outputs = self.unet(inputs, torch.tensor([0], device=get_device()), None)
        fine_shape = tuple(s * self.downscale_factor for s in self.coarse_shape)
        return outputs[..., : fine_shape[0], : fine_shape[1]]


@dataclasses.dataclass
class UNetRegressionSong:
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
        fine_topography: Optional[torch.Tensor],
    ):
        divisor = 2 ** (len(self.channel_mult) - 1)
        target_height, target_width = [
            s * downscale_factor
            + compute_unet_padding_size(s * downscale_factor, divisor)
            for s in coarse_shape
        ]
        unet = SongUNet(
            min(target_height, target_width),
            n_in_channels if fine_topography is None else n_in_channels + 1,
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
        return UNetRegressionModule(
            unet,
            coarse_shape,
            (target_height, target_width),
            downscale_factor,
            fine_topography,
        )


@dataclasses.dataclass
class UNetRegressionDhariwal:
    model_channels: int = 192
    channel_mult: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4])

    channel_mult_emb: int = 4
    num_blocks: int = 3
    attn_resolutions: List[int] = dataclasses.field(default_factory=lambda: [32, 16, 8])
    dropout: float = 0.10
    label_dropout: int = 0

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
        unet = DhariwalUNet(
            min(target_height, target_width),
            n_in_channels if fine_topography is None else n_in_channels + 1,
            n_out_channels,
            model_channels=self.model_channels,
            channel_mult=self.channel_mult,
            channel_mult_emb=self.channel_mult_emb,
            num_blocks=self.num_blocks,
            attn_resolutions=self.attn_resolutions,
            dropout=self.dropout,
            label_dropout=self.label_dropout,
        )
        return UNetRegressionModule(
            unet,
            coarse_shape,
            (target_height, target_width),
            downscale_factor,
            fine_topography,
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


@dataclasses.dataclass
class PreBuiltBuilder:
    module: torch.nn.Module

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        coarse_shape: Tuple[int, int],
        downscale_factor: int,
        fine_topography: Optional[torch.Tensor] = None,
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
        fine_topography: Optional[torch.Tensor] = None,
    ) -> torch.nn.Module:
        return Interpolate(downscale_factor, self.mode)


NET_REGISTRY: Mapping[str, Type[ModuleConfig]] = {
    "swinir": SwinirConfig,
    "prebuilt": PreBuiltBuilder,
    "interpolate": InterpolateConfig,
    "unet_regression_song": UNetRegressionSong,
    "unet_regression_dhariwal": UNetRegressionDhariwal,
}
