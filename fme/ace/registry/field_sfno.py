import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.dataset_info import DatasetInfo
from fme.core.models.conditional_sfno.fieldsfno import (
    FieldSFNONetConfig,
    get_lat_lon_fieldsfno,
)
from fme.core.models.conditional_sfno.layers import ContextConfig


@ModuleSelector.register("NoiseConditionedFieldSFNO")
@dataclasses.dataclass
class NoiseConditionedFieldSFNOBuilder(ModuleConfig):
    """
    Configuration for a noise-conditioned field-space SFNO model.

    The network's residual stream is the field itself: there is no encoder,
    decoder, or latent embedding. Forcing inputs condition the layer norms
    (alongside noise, labels, and the positional embedding) instead of being
    part of the residual stream, and diagnostic outputs start from learned
    initial values and are evolved like prognostic fields.

    The model is the identity on the prognostic fields at initialization, so
    it already predicts persistence. Use it with the step's
    ``residual_prediction`` off; with it on, the step would predict twice the
    input at initialization.

    Channel contract: the step passes channels in config order, so the step's
    ``in_names`` must list the prognostic names first (in the same order as
    ``out_names``) followed by the forcing names, and ``out_names`` must list
    the prognostic names first followed by the diagnostic names. Any extra
    input channels the step packs after ``in_names`` are treated as forcing.
    The builder checks the channel counts against ``n_prognostic_channels``
    but cannot check the names.

    Attributes:
        n_prognostic_channels: Number of prognostic channels, i.e. names in
            both ``in_names`` and ``out_names``.
        filter_type: Type of spectral filter to use.
        embed_dim: Base width of the block branches: the MLP hidden layer is
            ``int(embed_dim * mlp_ratio)`` wide and the spectral filter runs
            on ``round(embed_dim * spectral_ratio)`` channels, independent of
            the number of fields.
        spectral_ratio: Fraction of ``embed_dim`` the spectral filter runs
            on; each block projects the field to that width with a 1x1
            convolution before the filter and back after it.
        noise_embed_dim: Dimension of the noise embedding.
        noise_type: Type of noise to use for conditioning.
        context_pos_embed_dim: Dimension of the position embedding to use
            for conditioning.
        label_embed_dim: Dimension of the learned label embedding space.
            When > 0, a shared linear layer maps one-hot labels to this
            embedding dimension before downstream conditioning layers.
            When 0 (default), one-hot labels are used directly.
        forcing_embed_dim: Width of the forcing embedding that conditions
            the layer norms.
        forcing_encoder_layers: Number of 1x1 convolution layers in the
            forcing encoder.
        global_layer_norm: Whether to reduce along the spatial domain when
            applying layer normalization.
        num_layers: Number of blocks (spectral filter and MLP) in the model.
        use_mlp: Whether to use an MLP in each block.
        mlp_ratio: Ratio of the MLP hidden dimension to ``embed_dim``.
        activation_function: Activation function to use.
        drop_rate: Dropout rate inside the MLPs.
        drop_path_rate: Stochastic depth rate.
        hard_thresholding_fraction: Fraction of spectral modes to retain.
        checkpointing: Gradient checkpointing level.
        data_grid: Grid type for spherical harmonic transforms.
        filter_residual: Whether to filter each block's inner skip through a
            SHT round-trip.
        block_inner_skip: Whether each block adds a learned 1x1 convolution
            of the spectral filter's input to its output before the
            activation (the standard SFNO always does).
        block_mlp_input: What each block's MLP reads: ``"field"`` (the
            field after the filter branch's gated update; two updates per
            block) or ``"filter"`` (the filter branch's output directly;
            one gated update at the end of the block). Ignored when
            ``use_mlp`` is False.
        filter_output: Whether to filter the model's update (output minus
            input state) through a SHT round-trip.
        local_blocks: List of block indices to use discrete-continuous
            convolution (DISCO) blocks, which apply local filters.
        affine_norms: Whether to use element-wise affine parameters in the
            normalization layers.
        filter_num_groups: Number of groups to use in grouped convolutions
            for the spectral filter. Must divide the spectral filter width
            ``round(embed_dim * spectral_ratio)``.
        filter_preserves_global_mean: If True, the spectral filter preserves
            the l=0 (global mean) spherical harmonic coefficient.
        gamma_init: Initial value of each block's residual gates; 1e-12
            (the FieldSpaceNN default) makes the model the identity at init.
    """

    n_prognostic_channels: int
    filter_type: Literal["linear", "makani-linear"] = "linear"
    embed_dim: int = 256
    spectral_ratio: float = 1.0
    noise_embed_dim: int = 256
    noise_type: Literal["isotropic", "gaussian"] = "gaussian"
    context_pos_embed_dim: int = 0
    label_embed_dim: int = 0
    forcing_embed_dim: int = 32
    forcing_encoder_layers: int = 2
    global_layer_norm: bool = False
    num_layers: int = 12
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    activation_function: str = "gelu"
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    hard_thresholding_fraction: float = 1.0
    checkpointing: int = 0
    # healpix not supported due to assumptions about number of spatial dims
    data_grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
    filter_residual: bool = False
    block_inner_skip: bool = True
    block_mlp_input: Literal["field", "filter"] = "field"
    filter_output: bool = False
    local_blocks: list[int] | None = None
    affine_norms: bool = False
    filter_num_groups: int = 1
    filter_preserves_global_mean: bool = False
    gamma_init: float = 1e-12

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(state)

    def __post_init__(self):
        if self.n_prognostic_channels < 1:
            raise ValueError("n_prognostic_channels must be at least 1")
        # validate the network options at config load rather than at build
        self._net_config()

    def _net_config(self) -> FieldSFNONetConfig:
        return FieldSFNONetConfig(
            filter_type=self.filter_type,
            embed_dim=self.embed_dim,
            spectral_ratio=self.spectral_ratio,
            global_layer_norm=self.global_layer_norm,
            num_layers=self.num_layers,
            use_mlp=self.use_mlp,
            mlp_ratio=self.mlp_ratio,
            activation_function=self.activation_function,
            forcing_embed_dim=self.forcing_embed_dim,
            forcing_encoder_layers=self.forcing_encoder_layers,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            hard_thresholding_fraction=self.hard_thresholding_fraction,
            checkpointing=self.checkpointing,
            filter_num_groups=self.filter_num_groups,
            filter_residual=self.filter_residual,
            block_inner_skip=self.block_inner_skip,
            block_mlp_input=self.block_mlp_input,
            filter_output=self.filter_output,
            local_blocks=self.local_blocks,
            affine_norms=self.affine_norms,
            filter_preserves_global_mean=self.filter_preserves_global_mean,
            gamma_init=self.gamma_init,
        )

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ):
        if n_in_channels < self.n_prognostic_channels:
            raise ValueError(
                f"n_in_channels ({n_in_channels}) is smaller than "
                f"n_prognostic_channels ({self.n_prognostic_channels})"
            )
        if n_out_channels < self.n_prognostic_channels:
            raise ValueError(
                f"n_out_channels ({n_out_channels}) is smaller than "
                f"n_prognostic_channels ({self.n_prognostic_channels})"
            )
        n_labels = len(dataset_info.all_labels)
        if self.label_embed_dim > 0:
            effective_label_dim = self.label_embed_dim
        else:
            effective_label_dim = n_labels

        net = get_lat_lon_fieldsfno(
            params=self._net_config(),
            prognostic_chans=self.n_prognostic_channels,
            forcing_chans=n_in_channels - self.n_prognostic_channels,
            diagnostic_chans=n_out_channels - self.n_prognostic_channels,
            img_shape=dataset_info.img_shape,
            data_grid=self.data_grid,
            context_config=ContextConfig(
                embed_dim_scalar=0,
                embed_dim_pos=self.context_pos_embed_dim,
                embed_dim_noise=self.noise_embed_dim,
                embed_dim_labels=effective_label_dim,
            ),
        )
        inverse_sht: Callable[[torch.Tensor], torch.Tensor] | None
        if self.noise_type == "isotropic":
            inverse_sht = net.itrans_up
            # itrans_up is typed as nn.Module; lmax and mmax are its int
            # attributes, read via getattr so mypy accepts the int() call.
            lmax = int(getattr(net.itrans_up, "lmax"))
            mmax = int(getattr(net.itrans_up, "mmax"))
        else:
            inverse_sht = None
            lmax = 0
            mmax = 0
        return NoiseConditionedModel(
            net,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_pos=self.context_pos_embed_dim,
            n_labels=n_labels,
            label_embed_dim=self.label_embed_dim,
            img_shape=dataset_info.img_shape,
            inverse_sht=inverse_sht,
            lmax=lmax,
            mmax=mmax,
        )
