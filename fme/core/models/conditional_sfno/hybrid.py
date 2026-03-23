import dataclasses
from typing import Literal

import torch
import torch.nn as nn

from .ankur import AnkurLocalNetConfig, get_lat_lon_ankur_localnet
from .layers import Context, ContextConfig
from .localnet import LocalNetConfig, get_lat_lon_localnet
from .sfnonet import (
    SFNONetConfig,
    SphericalFourierNeuralOperatorNet,
    get_lat_lon_sfnonet,
)

LocalConfig = AnkurLocalNetConfig | LocalNetConfig


@dataclasses.dataclass
class HybridNetConfig:
    """Configuration for HybridNet.

    Attributes:
        backbone: Configuration for the SFNO backbone that produces
            prognostic output.
        local: Configuration for the local network that produces
            diagnostic output. Use ``AnkurLocalNetConfig`` (type="ankur")
            or ``LocalNetConfig`` (type="localnet").
        learn_residual: Whether to add the prognostic input directly
            to the prognostic output (identity skip connection).
        data_grid: Grid type for spherical harmonic transforms used
            by the SFNO backbone.
    """

    backbone: SFNONetConfig = dataclasses.field(default_factory=SFNONetConfig)
    local: LocalConfig = dataclasses.field(default_factory=AnkurLocalNetConfig)
    learn_residual: bool = False
    data_grid: Literal["legendre-gauss", "equiangular"] = "equiangular"


def get_lat_lon_hybridnet(
    params: HybridNetConfig,
    n_forcing_channels: int,
    n_prognostic_channels: int,
    n_diagnostic_channels: int,
    img_shape: tuple[int, int],
    embed_dim_labels: int = 0,
) -> "HybridNet":
    """Factory function to build a HybridNet.

    Args:
        params: HybridNet configuration.
        n_forcing_channels: Number of input-only (forcing) channels.
        n_prognostic_channels: Number of input-output (prognostic) channels.
        n_diagnostic_channels: Number of output-only (diagnostic) channels.
        img_shape: Spatial dimensions (lat, lon) of the input data.
        embed_dim_labels: Dimension of label embeddings for conditional
            layer normalization. 0 disables label conditioning.

    Returns:
        A configured HybridNet instance.
    """
    n_in = n_forcing_channels + n_prognostic_channels

    context_config = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_noise=0,
        embed_dim_labels=embed_dim_labels,
        embed_dim_pos=0,
    )

    backbone = get_lat_lon_sfnonet(
        params=params.backbone,
        in_chans=n_in,
        out_chans=n_prognostic_channels,
        img_shape=img_shape,
        data_grid=params.data_grid,
        context_config=context_config,
    )

    local_config = params.local
    if isinstance(local_config, AnkurLocalNetConfig):
        local_net: nn.Module = get_lat_lon_ankur_localnet(
            params=local_config,
            in_chans=n_in,
            out_chans=n_diagnostic_channels,
            img_shape=img_shape,
            data_grid=params.data_grid,
            context_config=context_config,
        )
    elif isinstance(local_config, LocalNetConfig):
        local_net = get_lat_lon_localnet(
            params=local_config,
            in_chans=n_in,
            out_chans=n_diagnostic_channels,
            img_shape=img_shape,
            data_grid=params.data_grid,
            context_config=context_config,
        )
    else:
        raise ValueError(f"Unknown local config type: {type(local_config)}")

    return HybridNet(
        backbone=backbone,
        local_net=local_net,
        learn_residual=params.learn_residual,
        n_prognostic_channels=n_prognostic_channels,
        embed_dim_labels=embed_dim_labels,
    )


class HybridNet(nn.Module):
    """Hybrid network combining an SFNO backbone with a local diagnostic network.

    Analogous to Ankur's ColumnDiagnosticSphericalFourierNeuralOperatorNet,
    but using the conditional SFNO and local networks compositionally.

    The SFNO backbone processes the concatenated forcing and prognostic input
    to produce a prognostic output. The local network processes the same
    concatenated input to produce a diagnostic output.

    Args:
        backbone: SFNO network for prognostic prediction.
        local_net: Local network for diagnostic prediction.
        learn_residual: Whether to add the prognostic input to the
            backbone output (identity residual connection).
        n_prognostic_channels: Number of prognostic channels, used
            to slice the input when learn_residual is True.
        embed_dim_labels: Dimension of label embeddings. 0 means no labels.
    """

    def __init__(
        self,
        backbone: SphericalFourierNeuralOperatorNet,
        local_net: nn.Module,
        learn_residual: bool = False,
        n_prognostic_channels: int = 0,
        embed_dim_labels: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.local_net = local_net
        self.learn_residual = learn_residual
        self.n_prognostic_channels = n_prognostic_channels
        self.embed_dim_labels = embed_dim_labels

    def forward(
        self,
        forcing: torch.Tensor,
        prognostic: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([forcing, prognostic], dim=1)

        context = Context(
            embedding_scalar=None,
            embedding_pos=None,
            labels=labels,
            noise=None,
        )

        prognostic_out = self.backbone(x, context)
        diagnostic_out = self.local_net(x, context)

        if self.learn_residual:
            prognostic_out = prognostic_out + prognostic

        return prognostic_out, diagnostic_out
