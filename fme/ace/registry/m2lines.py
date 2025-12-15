import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

from fme.ace.models.graphcast import GRAPHCAST_AVAIL
from fme.ace.models.graphcast.main import GraphCast
from fme.ace.models.ocean.m2lines.samudra import Samudra
from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.dataset_info import DatasetInfo


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
