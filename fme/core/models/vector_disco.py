"""VectorDiscoNetwork: a learned spherical NN built from VectorDiscoBlocks.

Stacks VectorDiscoBlocks to learn scalar-vector dynamics on the sphere.
Each block performs horizontal mixing (DISCO convolution), vertical mixing
(pointwise MLP across channels that include multiple levels), and nonlinear
scalar-vector coupling (ScalarVectorProduct).

The network is level-agnostic: levels are simply extra channels. The caller
is responsible for packing per-level and surface fields into scalar/vector
tensors and unpacking the outputs.
"""

import dataclasses

import torch
import torch.nn as nn

from fme.core.shallow_water.block import PointwiseVectorTransform, VectorDiscoBlock


@dataclasses.dataclass
class VectorDiscoNetworkConfig:
    """Configuration for a VectorDiscoNetwork.

    Attributes:
        n_scalar_channels: Total scalar latent channels in each block.
        n_vector_channels: Total vector latent channels in each block.
        n_blocks: Number of stacked VectorDiscoBlocks.
        kernel_shape: DISCO filter kernel shape.
        mlp_hidden_factor: Scalar MLP hidden dim = n_scalar_channels * this.
        activation: Activation function name ("gelu", "relu", "none").
        residual_blocks: Whether VectorDiscoBlocks use internal residuals.
    """

    n_scalar_channels: int = 64
    n_vector_channels: int = 16
    n_blocks: int = 4
    kernel_shape: int = 5
    mlp_hidden_factor: int = 2
    activation: str = "gelu"
    residual_blocks: bool = True


class VectorDiscoNetwork(nn.Module):
    """Learned spherical network built from stacked VectorDiscoBlocks.

    Input/output signature::

        (scalars: (B, N_in_s, H, W), vectors: (B, N_in_v, H, W, 2))
        → (scalars: (B, N_out_s, H, W), vectors: (B, N_out_v, H, W, 2))

    Architecture:
        1. Scalar encoder: Conv2d(N_in_s, n_scalar_channels, 1)
        2. Vector encoder: PointwiseVectorTransform(N_in_v, n_vector_channels)
        3. N × VectorDiscoBlock (horizontal+vertical mixing, sv_product)
        4. Scalar decoder: Conv2d(n_scalar_channels, N_out_s, 1)
        5. Vector decoder: PointwiseVectorTransform(n_vector_channels, N_out_v)
    """

    def __init__(
        self,
        config: VectorDiscoNetworkConfig,
        n_in_scalars: int,
        n_out_scalars: int,
        n_in_vectors: int,
        n_out_vectors: int,
        img_shape: tuple[int, int],
    ):
        super().__init__()
        Ns = config.n_scalar_channels
        Nv = config.n_vector_channels

        # Encoder
        self.scalar_encoder = nn.Conv2d(n_in_scalars, Ns, 1)
        self.vector_encoder = PointwiseVectorTransform(n_in_vectors, Nv)

        # Blocks
        self.blocks = nn.ModuleList(
            [
                VectorDiscoBlock(
                    n_scalar=Ns,
                    n_vector=Nv,
                    shape=img_shape,
                    kernel_shape=config.kernel_shape,
                    mlp_hidden_factor=config.mlp_hidden_factor,
                    activation=config.activation,
                    residual=config.residual_blocks,
                )
                for _ in range(config.n_blocks)
            ]
        )

        # Decoder
        self.scalar_decoder = nn.Conv2d(Ns, n_out_scalars, 1)
        self.vector_decoder = PointwiseVectorTransform(Nv, n_out_vectors)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for well-behaved forward pass.

        Zero-initializes:
        - Decoder weights (so initial network output is zero, enabling
          residual prediction to start as identity)
        - sv_product weights in each block (multiplicative term; random
          init causes variance explosion through stacked blocks)
        - MLP output layer in each block (residual branch; zero means
          MLP initially contributes nothing)
        """
        nn.init.zeros_(self.scalar_decoder.weight)
        if self.scalar_decoder.bias is not None:
            nn.init.zeros_(self.scalar_decoder.bias)
        nn.init.zeros_(self.vector_decoder.weight)

        for block in self.blocks:
            if block.sv_product is not None:
                nn.init.zeros_(block.sv_product.weight)
            if block.pointwise_vv is not None:
                nn.init.zeros_(block.pointwise_vv.weight)
            if block.scalar_mlp is not None:
                # Zero the output layer of the MLP (last Conv2d)
                for module in reversed(list(block.scalar_mlp.modules())):
                    if isinstance(module, nn.Conv2d):
                        nn.init.zeros_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                        break

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalars: (B, N_in_s, H, W)
            vectors: (B, N_in_v, H, W, 2)

        Returns:
            scalar_out: (B, N_out_s, H, W)
            vector_out: (B, N_out_v, H, W, 2)
        """
        s = self.scalar_encoder(scalars)
        v = self.vector_encoder(vectors)

        for block in self.blocks:
            s, v = block(s, v)

        return self.scalar_decoder(s), self.vector_decoder(v)
