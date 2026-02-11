import math
from collections.abc import Callable
from typing import Literal

import torch

from fme.core.dataset_info import DatasetInfo

from .sfnonet import Context, ContextConfig, get_lat_lon_sfnonet
from .sfnonet import SphericalFourierNeuralOperatorNet as ConditionalSFNO


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


def build(
    params,
    n_in_channels: int,
    n_out_channels: int,
    dataset_info: DatasetInfo,
):
    sfno_net = get_lat_lon_sfnonet(
        params=params,
        in_chans=n_in_channels,
        out_chans=n_out_channels,
        img_shape=dataset_info.img_shape,
        context_config=ContextConfig(
            embed_dim_scalar=0,
            embed_dim_pos=params.context_pos_embed_dim,
            embed_dim_noise=params.noise_embed_dim,
            embed_dim_labels=len(dataset_info.all_labels),
        ),
    )
    return NoiseConditionedSFNO(
        sfno_net,
        noise_type=params.noise_type,
        embed_dim_noise=params.noise_embed_dim,
        embed_dim_pos=params.context_pos_embed_dim,
        embed_dim_labels=len(dataset_info.all_labels),
        img_shape=dataset_info.img_shape,
    )
