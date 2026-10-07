import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

from fme.ace.models.graphcast import GRAPHCAST_AVAIL
from fme.ace.models.graphcast.main import GraphCast
from fme.ace.models.ocean.m2lines.layers import LatPad
from fme.ace.models.ocean.m2lines.samudra import ConditionedBlocks, Samudra
from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.dataset_info import DatasetInfo
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
        lat_pad: Padding of the latitude axis wherever the network pads it
            (block convolutions, final convolution, decoder skip alignment,
            and the upsampling when ``zonally_periodic_upsample`` is set).
            "constant" (the default, the original behavior) pads zeros,
            "reflect" mirrors about the edge row, and "pole" pads across the
            pole: the rows beyond a pole are the edge rows flipped in latitude
            and rotated by half the longitudes, the true neighbors of a scalar
            field on a grid whose edge cells touch the poles.
        pad_to_pool_multiple: Pad the latitude axis (with ``lat_pad``) up to a
            multiple of ``2 ** len(ch_width)`` before the U-Net and crop the
            output back, so pooling never drops a row at an odd height (180
            rows pad to 192 with the default four levels). The padding is
            split evenly between the edges, any odd extra row going at the end
            of the axis (the north edge for south-to-north latitude).

        Neither ``lat_pad`` nor ``pad_to_pool_multiple`` adds parameters, so a
        checkpoint trained without them can be fine-tuned with them on.
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
    pad_to_pool_multiple: bool = False
    noise_embed_dim: int = 0
    conditioned_blocks: ConditionedBlocks | None = None

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

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ):
        if len(dataset_info.all_labels) > 0:
            raise ValueError("Samudra does not support labels")
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
            pad_to_pool_multiple=self.pad_to_pool_multiple,
            context_config=context_config,
            conditioned_blocks=self.conditioned_blocks,
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
