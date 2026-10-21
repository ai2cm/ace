import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

from fme.ace.models.graphcast import GRAPHCAST_AVAIL
from fme.ace.models.graphcast.main import GraphCast
from fme.ace.models.ocean.m2lines.layers import LatPad
from fme.ace.models.ocean.m2lines.samudra import (
    ConditionedBlocks,
    Samudra,
    validate_exact_pooling_grid,
    validate_pooling_factors,
)
from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.dataset_info import DatasetInfo, MissingDatasetInfo
from fme.core.models.conditional_sfno.layers import ContextConfig


@ModuleSelector.register("Samudra")
@dataclasses.dataclass
class SamudraBuilder(ModuleConfig):
    """
    Configuration for the M2Lines Samudra architecture.

    Setting ``noise_embed_dim`` above zero makes the network noise-conditioned,
    zero makes it deterministic.

    Parameters:
        noise_embed_dim: Number of noise channels drawn and projected onto each
            conditioned block's scale and bias. Zero (the default) builds the
            deterministic network.
        conditioned_blocks: Which ConvNeXt blocks are conditioned. Required when
            ``noise_embed_dim`` is non-zero, and must be left None (the default)
            when it is zero, where there is nothing to condition on.
            "bottleneck" conditions only the block at the coarsest resolution.
            "all_blocks" conditions every block and reaches the finest scales.
        norm: Normalization used inside each ConvNeXt block. This choice
            interacts with noise conditioning, which is applied as a FiLM scale
            and bias after the norm: after "layer" norm the network can modulate
            the conditioning strength per sample, while after "instance" norm
            that strength is a learned constant, identical for every sample on
            every step. "layer" is the principled choice for a conditioned
            network.
        lat_pad: Latitude padding mode: "constant" (default, zeros) or
            "pole" (continues across the pole). See ``Samudra``.
        pad_pool: Pad odd sizes before each pool instead of dropping the last
            row or column. See ``Samudra``.
        pooling: Pooling factor of each U-Net level, one per entry of
            ``ch_width``. When given, the product of the factors must divide
            both grid dimensions, so every level pools and upsamples exactly;
            ``build`` checks this against the dataset's grid. None (the
            default) pools by 2 at every level, floor-pooling grids 2 does not
            divide. See ``Samudra``.
        gelu_cap: Maximum of the GELU in each ConvNeXt block (default 1.0,
            the original network), or None for an uncapped GELU. See
            ``Samudra``.
        conservative_pool: Pool conservatively to ``ceil(n / 2)`` cells per
            axis with no padding, and upsample by corrected linear
            interpolation to each skip connection's shape. Takes neither
            ``pad_pool`` nor ``pooling``. See ``Samudra``.
    """

    ch_width: list[int] = dataclasses.field(
        default_factory=lambda: [200, 250, 300, 400]
    )
    n_layers: list[int] = dataclasses.field(default_factory=lambda: [1, 1, 1, 1])
    dilation: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8])
    pad: str = "circular"
    norm: Literal["batch", "instance", "layer"] = "instance"
    norm_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    upscale_factor: int = 4
    checkpoint_strategy: Literal["all", "simple"] | None = None
    zonally_periodic_upsample: bool = False
    lat_pad: LatPad = "constant"
    pad_pool: bool = False
    gelu_cap: float | None = 1.0
    conservative_pool: bool = False
    noise_embed_dim: int = 0
    conditioned_blocks: ConditionedBlocks | None = None
    pooling: list[int] | None = None

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(state)

    def __post_init__(self):
        if "num_features" in self.norm_kwargs:
            raise ValueError("norm_kwargs should not have num_features")
        if "normalized_shape" in self.norm_kwargs:
            raise ValueError("norm_kwargs should not have normalized_shape")
        if self.noise_embed_dim < 0:
            raise ValueError("noise_embed_dim must not be negative")
        if self.noise_embed_dim > 0 and self.conditioned_blocks is None:
            raise ValueError(
                "noise_embed_dim requires conditioned_blocks to be set; there is "
                "no default choice of where to inject noise."
            )
        if self.noise_embed_dim == 0 and self.conditioned_blocks is not None:
            raise ValueError(
                "conditioned_blocks requires a non-zero noise_embed_dim; without "
                "it the network is deterministic and conditions on nothing."
            )
        if self.pooling is not None:
            validate_pooling_factors(self.pooling, len(self.ch_width))
            if self.pad_pool:
                raise ValueError(
                    "pad_pool has nothing to pad under exact pooling factors"
                )
        if self.conservative_pool and (self.pad_pool or self.pooling is not None):
            raise ValueError(
                "conservative_pool replaces the pool, so it takes neither "
                "pad_pool nor exact pooling factors"
            )

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ):
        if len(dataset_info.all_labels) > 0:
            raise ValueError("Samudra does not support labels")
        if self.pooling is not None:
            try:
                img_shape = dataset_info.img_shape
            except MissingDatasetInfo:
                pass  # Samudra.forward checks the grid instead
            else:
                validate_exact_pooling_grid(self.pooling, img_shape, self.lat_pad)
        if self.noise_embed_dim > 0:
            context_config: ContextConfig | None = ContextConfig(
                embed_dim_scalar=0,
                embed_dim_labels=0,
                embed_dim_noise=self.noise_embed_dim,
                embed_dim_pos=0,
            )
        else:
            context_config = None
        samudra = Samudra(
            input_channels=n_in_channels,
            output_channels=n_out_channels,
            ch_width=self.ch_width,
            dilation=self.dilation,
            n_layers=self.n_layers,
            pad=self.pad,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            upscale_factor=self.upscale_factor,
            checkpoint_strategy=self.checkpoint_strategy,
            zonally_periodic_upsample=self.zonally_periodic_upsample,
            lat_pad=self.lat_pad,
            pad_pool=self.pad_pool,
            gelu_cap=self.gelu_cap,
            conservative_pool=self.conservative_pool,
            context_config=context_config,
            conditioned_blocks=self.conditioned_blocks,
            pooling=self.pooling,
        )
        if context_config is None:
            return samudra
        return NoiseConditionedModel(
            samudra,
            img_shape=dataset_info.img_shape,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_pos=0,
            n_labels=0,
            label_embed_dim=0,
        )


@ModuleSelector.register("FloeNet")
@dataclasses.dataclass
class FloeNetBuilder(ModuleConfig):
    """
    Configuration for the M2Lines FloeNet architecture.
    """

    latent_dimension: int = 256
    activation: str = "SiLU"
    meshes: int = 6
    M0: int = 4
    bias: bool = True
    radius_fraction: float = 1.0
    layernorm: bool = True
    processor_steps: int = 4
    residual: bool = True
    is_ocean: bool = True

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(state)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ):
        if not GRAPHCAST_AVAIL:
            raise ImportError("GraphCast dependencies (trimesh, rtree) not available.")
        return GraphCast(
            input_channels=n_in_channels,
            output_channels=n_out_channels,
            dataset_info=dataset_info,
            latent_dimension=self.latent_dimension,
            activation=self.activation,
            meshes=self.meshes,
            M0=self.M0,
            bias=self.bias,
            radius_fraction=self.radius_fraction,
            layernorm=self.layernorm,
            processor_steps=self.processor_steps,
            residual=self.residual,
            is_ocean=self.is_ocean,
        )
