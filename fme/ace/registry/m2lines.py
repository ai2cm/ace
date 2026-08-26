import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

from fme.ace.models.graphcast import GRAPHCAST_AVAIL
from fme.ace.models.graphcast.main import GraphCast
from fme.ace.models.ocean.m2lines.samudra import NoiseInjection, Samudra
from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.dataset_info import DatasetInfo
from fme.core.models.conditional_sfno.layers import ContextConfig


@ModuleSelector.register("Samudra")
@dataclasses.dataclass
class SamudraBuilder(ModuleConfig):
    """
    Configuration for the M2Lines Samudra architecture.
    """

    ch_width: list[int] = dataclasses.field(
        default_factory=lambda: [200, 250, 300, 400]
    )
    n_layers: list[int] = dataclasses.field(default_factory=lambda: [1, 1, 1, 1])
    dilation: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8])
    pad: str = "circular"
    norm: str = "instance"
    norm_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    upscale_factor: int = 4
    checkpoint_strategy: Literal["all", "simple"] | None = None
    zonally_periodic_upsample: bool = False

    def __post_init__(self):
        if "num_features" in self.norm_kwargs:
            raise ValueError("norm_kwargs should not have num_features")
        if "normalized_shape" in self.norm_kwargs:
            raise ValueError("norm_kwargs should not have normalized_shape")

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ):
        if len(dataset_info.all_labels) > 0:
            raise ValueError("Samudra does not support labels")
        return Samudra(
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
        )


@ModuleSelector.register("NoiseConditionedSamudra")
@dataclasses.dataclass
class NoiseConditionedSamudraBuilder(SamudraBuilder):
    """
    Configuration for a noise-conditioned M2Lines Samudra architecture.

    A noise field is drawn on every forward call and supplied as conditioning
    input to the ConvNeXt blocks' normalization layers, so an ensemble of
    members can be drawn from one input and trained against a proper scoring
    rule. The conditioning weights are zero-initialized, so an untrained model
    is exactly deterministic.

    Parameters:
        noise_embed_dim: Number of noise channels drawn and projected onto each
            conditioned block's scale and bias. Defaults to 32, DLESyM-Ocean's
            choice.
        noise_injection: Which ConvNeXt blocks are conditioned. "bottleneck"
            conditions only the block at the coarsest resolution (after the
            encoder's AvgPools), so the noise perturbs large scales;
            "all_blocks" conditions every block, the pattern the ACE SFNO uses,
            which also reaches the finest scales.
    """

    noise_embed_dim: int = 32
    noise_injection: NoiseInjection = "bottleneck"

    def __post_init__(self):
        super().__post_init__()
        if self.noise_embed_dim <= 0:
            raise ValueError("noise_embed_dim must be positive")
        if self.norm is None:
            raise ValueError(
                "noise conditioning needs a normalization layer to condition, "
                "so norm cannot be None"
            )

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ):
        if len(dataset_info.all_labels) > 0:
            raise ValueError("NoiseConditionedSamudra does not support labels")
        context_config = ContextConfig(
            embed_dim_scalar=0,
            embed_dim_labels=0,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_pos=0,
        )
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
            context_config=context_config,
            noise_injection=self.noise_injection,
        )
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
