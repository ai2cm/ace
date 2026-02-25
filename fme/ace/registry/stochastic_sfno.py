import dataclasses
import math
from collections.abc import Callable
from typing import Literal

import torch

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.dataset_info import DatasetInfo
from fme.core.models.conditional_sfno.sfnonet import (
    Context,
    ContextConfig,
    get_lat_lon_sfnonet,
)
from fme.core.models.conditional_sfno.sfnonet import (
    SphericalFourierNeuralOperatorNet as ConditionalSFNO,
)


def isotropic_noise(
    leading_shape: tuple[int, ...],
    lmax: int,  # length of the ℓ axis expected by isht
    mmax: int,  # length of the m axis expected by isht
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

    return isht(alm)


class NoiseConditionedSFNO(torch.nn.Module):
    def __init__(
        self,
        conditional_model: ConditionalSFNO,
        img_shape: tuple[int, int],
        noise_type: Literal["isotropic", "gaussian"] = "gaussian",
        embed_dim_noise: int = 256,
        embed_dim_pos: int = 0,
        embed_dim_labels: int = 0,
    ):
        super().__init__()
        self.conditional_model = conditional_model
        self.embed_dim = embed_dim_noise
        self.noise_type = noise_type
        self.label_pos_embed: torch.nn.Parameter | None = None
        # register pos embed if pos_embed_dim != 0
        if embed_dim_pos != 0:
            self.pos_embed = torch.nn.Parameter(
                torch.zeros(
                    1, embed_dim_pos, img_shape[0], img_shape[1], requires_grad=True
                )
            )
            # initialize pos embed with std=0.02
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if embed_dim_labels > 0:
                self.label_pos_embed = torch.nn.Parameter(
                    torch.zeros(
                        embed_dim_labels,
                        embed_dim_pos,
                        img_shape[0],
                        img_shape[1],
                        requires_grad=True,
                    )
                )
                torch.nn.init.trunc_normal_(self.label_pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x.reshape(-1, *x.shape[-3:])
        if self.noise_type == "isotropic":
            lmax = self.conditional_model.itrans_up.lmax
            mmax = self.conditional_model.itrans_up.mmax
            noise = isotropic_noise(
                (x.shape[0], self.embed_dim),
                lmax,
                mmax,
                self.conditional_model.itrans_up,
                device=x.device,
            )
        elif self.noise_type == "gaussian":
            noise = torch.randn(
                [x.shape[0], self.embed_dim, *x.shape[-2:]],
                device=x.device,
                dtype=x.dtype,
            )
        else:
            raise ValueError(f"Invalid noise type: {self.noise_type}")

        if self.pos_embed is not None:
            embedding_pos = self.pos_embed.repeat(noise.shape[0], 1, 1, 1)
            if self.label_pos_embed is not None and labels is not None:
                label_embedding_pos = torch.einsum(
                    "bl, lpxy -> bpxy", labels, self.label_pos_embed
                )
                embedding_pos = embedding_pos + label_embedding_pos
        else:
            embedding_pos = None

        return self.conditional_model(
            x,
            Context(
                embedding_scalar=None,
                embedding_pos=embedding_pos,
                labels=labels,
                noise=noise,
            ),
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
        spectral_transform: Type of spherical transform to use.
            Kept for backwards compatibility.
        filter_type: Type of filter to use.
        operator_type: Type of operator to use. Only "dhconv" is supported.
        residual_filter_factor: Factor by which to downsample the residual.
        embed_dim: Dimension of the embedding.
        noise_embed_dim: Dimension of the noise embedding.
        noise_type: Type of noise to use for conditioning.
        context_pos_embed_dim: Dimension of the position embedding to use
            for conditioning.
        global_layer_norm: Whether to reduce along the spatial domain when applying
            layer normalization.
        num_layers: Number of blocks (SFNO and MLP)in the model.
        use_mlp: Whether to use a MLP in the model.
        mlp_ratio: Ratio of the MLP hidden dimension
            to the embedding dimension.
        activation_function: Activation function to use.
        encoder_layers: Number of encoder layers in the model.
        pos_embed: Whether to use a position embedding.
        big_skip: Whether to use a big skip connection in the model.
        rank: Rank of the model.
        factorization: Unused, kept for backwards compatibility only.
        separable: Unused, kept for backwards compatibility only.
        complex_network: Whether to use a complex network.
        complex_activation: Activation function to use.
        spectral_layers: Number of spectral layers in the model.
        checkpointing: Whether to use checkpointing.
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
        sfno_net = get_lat_lon_sfnonet(
            params=self,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=dataset_info.img_shape,
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
