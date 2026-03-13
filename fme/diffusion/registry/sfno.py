import dataclasses
from typing import Literal

from fme.core.models.conditional_sfno.sfnonet import (
    ContextConfig,
    SFNONetConfig,
    get_lat_lon_sfnonet,
)
from fme.core.models.conditional_sfno.sfnonet import (
    SphericalFourierNeuralOperatorNet as ConditionalSFNO,
)

from .module import ModuleConfig, ModuleSelector


# this is based on the call signature of SphericalFourierNeuralOperatorNet at
# https://github.com/NVIDIA/modulus/blob/b8e27c5c4ebc409e53adaba9832138743ede2785/modulus/models/sfno/sfnonet.py#L292  # noqa: E501
@ModuleSelector.register("ConditionalSFNO")
@dataclasses.dataclass
class ConditionalSFNOBuilder(ModuleConfig):
    """
    Configuration for the SFNO architecture used in FourCastNet-SFNO.
    """

    spectral_transform: str = "sht"
    """Unused, kept for backwards compatibility only."""
    filter_type: str = "linear"
    operator_type: Literal["dhconv"] = "dhconv"
    """Unused, kept for backwards compatibility only. Must be "dhconv"."""
    scale_factor: int = 1
    embed_dim: int = 256
    num_layers: int = 12
    hard_thresholding_fraction: float = 1.0
    normalization_layer: str = "instance_norm"
    """Unused, kept for backwards compatibility only."""
    use_mlp: bool = True
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    big_skip: bool = True
    rank: float = 1.0
    """Unused, kept for backwards compatibility only."""
    factorization: str | None = None
    """Unused, kept for backwards compatibility only. Must be None."""
    separable: bool = False
    """Unused, kept for backwards compatibility only. Must be False."""
    complex_network: bool = True
    """Unused, kept for backwards compatibility only."""
    complex_activation: str = "real"
    """Unused, kept for backwards compatibility only."""
    spectral_layers: int = 1
    """Unused, kept for backwards compatibility only."""
    checkpointing: int = 0
    data_grid: Literal["legendre-gauss", "equiangular", "healpix"] = "legendre-gauss"

    def __post_init__(self):
        if self.factorization is not None:
            raise ValueError("The 'factorization' parameter is no longer supported.")
        if self.separable:
            raise ValueError("The 'separable' parameter is no longer supported.")
        if self.operator_type != "dhconv":
            raise ValueError(
                "Only 'dhconv' operator_type is supported for "
                "ConditionalSFNO models."
            )

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: tuple[int, int],
        n_sigma_embedding_channels: int,
    ) -> ConditionalSFNO:
        sfno_config = SFNONetConfig(
            embed_dim=self.embed_dim,
            filter_type=self.filter_type,
            scale_factor=self.scale_factor,
            num_layers=self.num_layers,
            hard_thresholding_fraction=self.hard_thresholding_fraction,
            use_mlp=self.use_mlp,
            activation_function=self.activation_function,
            encoder_layers=self.encoder_layers,
            pos_embed=self.pos_embed,
            big_skip=self.big_skip,
            checkpointing=self.checkpointing,
            data_grid=self.data_grid,
        )
        sfno_net = get_lat_lon_sfnonet(
            params=sfno_config,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=img_shape,
            context_config=ContextConfig(
                embed_dim_scalar=n_sigma_embedding_channels,
                embed_dim_labels=0,
                embed_dim_noise=0,
                embed_dim_pos=0,
            ),
        )

        return sfno_net
