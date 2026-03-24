import dataclasses
import math
from collections.abc import Callable
from typing import Literal

import torch

from fme.ace.registry.noise_conditioned import (
    NoiseConditionedModule,
    NoiseGenerator,
    gaussian_noise,
)
from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed.distributed import Distributed
from fme.core.models.conditional_sfno.sfnonet import (
    ContextConfig,
    SFNONetConfig,
    get_lat_lon_sfnonet,
)
from fme.core.models.conditional_sfno.sfnonet import (
    SphericalFourierNeuralOperatorNet as ConditionalSFNO,
)


def isotropic_noise(
    leading_shape: tuple[int, ...],
    lmax: int,  # length of the ℓ axis expected by isht (global)
    mmax: int,  # length of the m axis expected by isht (global)
    isht: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    # --- draw independent N(0,1) parts --------------------------------------
    coeff_shape = (*leading_shape, lmax, mmax)
    real = torch.randn(coeff_shape, dtype=torch.float32, device=device)
    imag = torch.randn(coeff_shape, dtype=torch.float32, device=device)
    imag[..., :, 0] = 0.0  # m = 0 ⇒ purely real

    # m > 0: make Re and Im each N(0,½)  → |a_{ℓ m}|² has variance 1
    sqrt2 = math.sqrt(2.0)
    real[..., :, 1:] /= sqrt2
    imag[..., :, 1:] /= sqrt2

    # --- global scale that makes Var[T(θ,φ)] = 1 ---------------------------
    scale = math.sqrt(4.0 * math.pi) / lmax  # (Unsöld theorem ⇒ L = lmax)
    alm = (real + 1j * imag) * scale

    # --- for distributed iSHT, slice to local spectral extent --------------
    l_slice, m_slice = Distributed.get_instance().get_local_slices((lmax, mmax))
    alm = alm[..., l_slice, m_slice]

    return isht(alm)


def _make_sfno_noise_generator(
    noise_type: Literal["isotropic", "gaussian"],
    conditional_model: ConditionalSFNO,
) -> NoiseGenerator:
    """Create a noise generator for an SFNO model.

    For gaussian noise, returns the default generator. For isotropic noise,
    returns a generator that uses the SFNO's inverse spherical harmonic
    transform.
    """
    if noise_type == "gaussian":
        return gaussian_noise
    elif noise_type == "isotropic":

        def _isotropic(x: torch.Tensor, embed_dim_noise: int) -> torch.Tensor:
            lmax = conditional_model.itrans_up.lmax
            mmax = conditional_model.itrans_up.mmax
            return isotropic_noise(
                (x.shape[0], embed_dim_noise),
                lmax,
                mmax,
                conditional_model.itrans_up,
                device=x.device,
            )

        return _isotropic
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")


def NoiseConditionedSFNO(
    conditional_model: ConditionalSFNO,
    img_shape: tuple[int, int],
    noise_type: Literal["isotropic", "gaussian"] = "gaussian",
    embed_dim_noise: int = 256,
    embed_dim_pos: int = 0,
    embed_dim_labels: int = 0,
) -> NoiseConditionedModule:
    """Create a noise-conditioned SFNO with support for isotropic noise."""
    return NoiseConditionedModule(
        module=conditional_model,
        img_shape=img_shape,
        embed_dim_noise=embed_dim_noise,
        embed_dim_pos=embed_dim_pos,
        embed_dim_labels=embed_dim_labels,
        noise_generator=_make_sfno_noise_generator(noise_type, conditional_model),
    )


# this is based on the call signature of SphericalFourierNeuralOperatorNet at
# https://github.com/NVIDIA/modulus/blob/b8e27c5c4ebc409e53adaba9832138743ede2785/modulus/models/sfno/sfnonet.py#L292  # noqa: E501
@ModuleSelector.register("NoiseConditionedSFNO")
@dataclasses.dataclass
class NoiseConditionedSFNOBuilder(ModuleConfig):
    """
    Configuration for a noise-conditioned SFNO model.

    Noise is provided as conditioning input to conditional layer normalization.

    Attributes:
        spectral_transform: Unused, kept for backwards compatibility only.
        filter_type: Type of filter to use.
        operator_type: Unused, kept for backwards compatibility only.
            Must be "dhconv".
        residual_filter_factor: Factor by which to downsample the residual.
        embed_dim: Dimension of the embedding.
        noise_embed_dim: Dimension of the noise embedding.
        noise_type: Type of noise to use for conditioning.
        context_pos_embed_dim: Dimension of the position embedding to use
            for conditioning.
        global_layer_norm: Whether to reduce along the spatial domain when applying
            layer normalization.
        num_layers: Number of blocks (SFNO and MLP) in the model.
        use_mlp: Whether to use an MLP in the model.
        mlp_ratio: Ratio of the MLP hidden dimension
            to the embedding dimension.
        activation_function: Activation function to use.
        encoder_layers: Number of encoder layers in the model.
        pos_embed: Whether to use a position embedding.
        big_skip: Whether to use a big skip connection in the model.
        rank: Unused, kept for backwards compatibility only.
        factorization: Unused, kept for backwards compatibility only.
            Must be None.
        separable: Unused, kept for backwards compatibility only.
            Must be False.
        complex_network: Unused, kept for backwards compatibility only.
        complex_activation: Unused, kept for backwards compatibility only.
        spectral_layers: Unused, kept for backwards compatibility only.
        checkpointing: Whether to use checkpointing.
        data_grid: Grid type for spherical harmonic transforms.
        filter_residual: Whether to filter residual connections through a
            SHT round-trip. These will always be filtered if residual_filter_factor
            is not 1.
        filter_output: Whether to filter the output of the model through a
            SHT round-trip.
        local_blocks: List of block indices to use discrete-conditional
            convolution (DISCO) blocks, which apply local filters. See
            Ocampo et al. (2022)
            https://arxiv.org/abs/2209.13603 for more details.
        normalize_big_skip: Whether to normalize the big_skip connection.
        affine_norms: Whether to use element-wise affine parameters in the
            normalization layers.
        filter_num_groups: Number of groups to use in grouped convolutions
            for the spectral filter.
        lora_rank: Rank of the LoRA adaptations outside of spectral convolutions.
            0 (default) disables LoRA.
        lora_alpha: Strength of the LoRA adaptations outside of spectral convolutions.
            Defaults to lora_rank.
        spectral_lora_rank: Rank of the LoRA adaptations for spectral convolutions.
            0 (default) disables LoRA.
        spectral_lora_alpha: Strength of the LoRA adaptations for spectral convolutions.
            Defaults to spectral_lora_rank.
    """

    spectral_transform: Literal["sht"] = "sht"
    filter_type: Literal["linear", "makani-linear"] = "linear"
    operator_type: Literal["dhconv"] = "dhconv"
    residual_filter_factor: int = 1
    embed_dim: int = 256
    noise_embed_dim: int = 256
    context_pos_embed_dim: int = 0
    noise_type: Literal["isotropic", "gaussian"] = "gaussian"
    global_layer_norm: bool = False
    num_layers: int = 12
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    big_skip: bool = True
    rank: float = 1.0
    factorization: None = None
    separable: bool = False
    complex_network: bool = True
    complex_activation: str = "real"
    spectral_layers: int = 1
    checkpointing: int = 0
    # healpix not supported due to assumptions about number of spatial dims
    data_grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
    filter_residual: bool = False
    filter_output: bool = False
    local_blocks: list[int] | None = None
    normalize_big_skip: bool = False
    affine_norms: bool = False
    filter_num_groups: int = 1
    lora_rank: int = 0
    lora_alpha: float | None = None
    spectral_lora_rank: int = 0
    spectral_lora_alpha: float | None = None

    def __post_init__(self):
        if self.context_pos_embed_dim > 0 and self.pos_embed:
            raise ValueError(
                "context_pos_embed_dim and pos_embed should not both be set"
            )
        if self.factorization is not None:
            raise ValueError("The 'factorization' parameter is no longer supported.")
        if self.separable:
            raise ValueError("The 'separable' parameter is no longer supported.")
        if self.operator_type != "dhconv":
            raise ValueError(
                "Only 'dhconv' operator_type is supported for "
                "NoiseConditionedSFNO models."
            )

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ):
        sfno_config = SFNONetConfig(
            embed_dim=self.embed_dim,
            filter_type=self.filter_type,
            global_layer_norm=self.global_layer_norm,
            num_layers=self.num_layers,
            use_mlp=self.use_mlp,
            mlp_ratio=self.mlp_ratio,
            activation_function=self.activation_function,
            encoder_layers=self.encoder_layers,
            pos_embed=self.pos_embed,
            big_skip=self.big_skip,
            checkpointing=self.checkpointing,
            filter_residual=self.filter_residual,
            filter_output=self.filter_output,
            local_blocks=self.local_blocks,
            normalize_big_skip=self.normalize_big_skip,
            affine_norms=self.affine_norms,
            filter_num_groups=self.filter_num_groups,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            spectral_lora_rank=self.spectral_lora_rank,
            spectral_lora_alpha=self.spectral_lora_alpha,
        )
        sfno_net = get_lat_lon_sfnonet(
            params=sfno_config,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=dataset_info.img_shape,
            data_grid=self.data_grid,
            context_config=ContextConfig(
                embed_dim_scalar=0,
                embed_dim_pos=self.context_pos_embed_dim,
                embed_dim_noise=self.noise_embed_dim,
                embed_dim_labels=len(dataset_info.all_labels),
            ),
        )
        return NoiseConditionedSFNO(
            sfno_net,
            noise_type=self.noise_type,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_pos=self.context_pos_embed_dim,
            embed_dim_labels=len(dataset_info.all_labels),
            img_shape=dataset_info.img_shape,
        )
