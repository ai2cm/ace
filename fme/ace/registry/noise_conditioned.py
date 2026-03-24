"""Noise-conditioned wrapper for modules with a Context-based forward signature."""

from collections.abc import Callable

import torch
from torch import nn

from fme.core.distributed import Distributed
from fme.core.models.conditional_sfno.layers import Context

NoiseGenerator = Callable[[torch.Tensor, int], torch.Tensor]
"""Callable (x, embed_dim_noise) -> noise_tensor.

Takes the input tensor (for shape/device/dtype) and noise embedding dimension,
returns noise of shape [x.shape[0], embed_dim_noise, *x.shape[-2:]].
"""


def gaussian_noise(x: torch.Tensor, embed_dim_noise: int) -> torch.Tensor:
    return torch.randn(
        [x.shape[0], embed_dim_noise, *x.shape[-2:]],
        device=x.device,
        dtype=x.dtype,
    )


class NoiseConditionedModule(nn.Module):
    """Wraps a context-based module with noise conditioning.

    Generates noise and optional positional embeddings (with label-position
    interaction), then calls the wrapped module with a fully populated Context.

    Args:
        module: An nn.Module with forward signature (x, context: Context).
        img_shape: Global spatial dimensions (lat, lon) of the input data.
        embed_dim_noise: Dimension of noise channels.
        embed_dim_pos: Dimension of learned positional embedding. 0 disables.
        embed_dim_labels: Dimension of label embeddings. 0 disables.
        noise_generator: Callable that produces noise given the input tensor
            and noise embedding dimension. Defaults to gaussian noise.
    """

    def __init__(
        self,
        module: nn.Module,
        img_shape: tuple[int, int],
        embed_dim_noise: int = 256,
        embed_dim_pos: int = 0,
        embed_dim_labels: int = 0,
        noise_generator: NoiseGenerator = gaussian_noise,
    ):
        super().__init__()
        self.module = module
        self.embed_dim_noise = embed_dim_noise
        self.img_shape = img_shape
        self._noise_generator = noise_generator
        self.label_pos_embed: nn.Parameter | None = None
        if embed_dim_pos != 0:
            self.pos_embed: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    1, embed_dim_pos, img_shape[0], img_shape[1], requires_grad=True
                )
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if embed_dim_labels > 0:
                self.label_pos_embed = nn.Parameter(
                    torch.zeros(
                        embed_dim_labels,
                        embed_dim_pos,
                        img_shape[0],
                        img_shape[1],
                        requires_grad=True,
                    )
                )
                nn.init.trunc_normal_(self.label_pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x.reshape(-1, *x.shape[-3:])
        noise = self._noise_generator(x, self.embed_dim_noise)

        h_slice, w_slice = Distributed.get_instance().get_local_slices(self.img_shape)

        embedding_pos: torch.Tensor | None = None
        if self.pos_embed is not None:
            pos_local = self.pos_embed[..., h_slice, w_slice]
            embedding_pos = pos_local.repeat(x.shape[0], 1, 1, 1)
            if self.label_pos_embed is not None and labels is not None:
                label_local = self.label_pos_embed[..., h_slice, w_slice]
                label_embedding_pos = torch.einsum(
                    "bl, lpxy -> bpxy", labels, label_local
                )
                embedding_pos = embedding_pos + label_embedding_pos

        return self.module(
            x,
            Context(
                embedding_scalar=None,
                embedding_pos=embedding_pos,
                labels=labels,
                noise=noise,
            ),
        )
