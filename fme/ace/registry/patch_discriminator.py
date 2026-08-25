import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.dataset_info import DatasetInfo


def _pad_sphere(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Circular padding in longitude, zero padding in latitude."""
    x = F.pad(x, (pad, pad, 0, 0), mode="circular")
    x = F.pad(x, (0, 0, pad, pad), mode="constant", value=0)
    return x


@ModuleSelector.register("PatchDiscriminator")
@dataclasses.dataclass
class PatchDiscriminatorConfig(ModuleConfig):
    """
    Configuration for a small convolutional patch discriminator.

    Applies two 3x3 convolutions with circular longitude padding and zero
    latitude padding (barrier condition at the poles), followed by a 1x1
    projection. All convolutions use spectral normalization. Two positional
    channels (sin(lat), cos(lat)) are concatenated to the input.

    Parameters:
        hidden_dim: First hidden layer width; the second layer doubles it.
    """

    hidden_dim: int = 64

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        return PatchDiscriminator(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            hidden_dim=self.hidden_dim,
            lat=dataset_info.horizontal_coordinates.lat_1d,
            n_lon=dataset_info.img_shape[1],
        )


class PatchDiscriminator(nn.Module):
    """Two-layer 3x3 CNN with spectral normalization and positional encoding.

    The forward pass prepends sin(lat) and cos(lat) as two extra input
    channels, then runs::

        PadSphere(1) -> Conv3x3 -> LeakyReLU(0.2)
        PadSphere(1) -> Conv3x3 -> LeakyReLU(0.2)
        Conv1x1

    Output has the same spatial shape as the input: longitude is circular-
    padded, latitude is zero-padded (barrier condition at the poles).
    """

    def __init__(
        self,
        n_in_channels: int,
        n_out_channels: int,
        hidden_dim: int,
        lat: torch.Tensor,
        n_lon: int,
    ):
        super().__init__()
        # Positional encoding buffers: (1, 2, n_lat, n_lon)
        lat_rad = lat.float()
        sin_lat = torch.sin(lat_rad).unsqueeze(0).unsqueeze(0)  # (1, 1, n_lat)
        cos_lat = torch.cos(lat_rad).unsqueeze(0).unsqueeze(0)  # (1, 1, n_lat)
        pos = torch.cat([sin_lat, cos_lat], dim=1)  # (1, 2, n_lat)
        pos = pos.unsqueeze(-1).expand(-1, -1, -1, n_lon).contiguous()
        self.register_buffer("pos_encoding", pos)

        ch_in = n_in_channels + 2  # +2 for positional channels
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(ch_in, hidden_dim, kernel_size=3, padding=0)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=0)
        )
        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2d(hidden_dim * 2, n_out_channels, kernel_size=1)
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate positional encoding
        pos = self.pos_encoding.expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, pos], dim=1)

        x = _pad_sphere(x, 1)
        x = self.act(self.conv1(x))

        x = _pad_sphere(x, 1)
        x = self.act(self.conv2(x))

        x = self.conv3(x)
        return x
