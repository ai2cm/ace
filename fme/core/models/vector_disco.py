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

from fme.core.models.conditional_sfno.layers import Context, ContextConfig
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
        context: Conditioning dimensions for the post-norm in each block.
            Supports noise injection and positional embedding. When all
            embed dims are 0 (default), the norm is unconditional.
    """

    n_scalar_channels: int = 64
    n_vector_channels: int = 16
    n_blocks: int = 4
    kernel_shape: int = 5
    mlp_hidden_factor: int = 2
    activation: str = "gelu"
    residual_blocks: bool = True
    num_groups: int = 1
    context: ContextConfig = dataclasses.field(
        default_factory=lambda: ContextConfig(
            embed_dim_scalar=0,
            embed_dim_labels=0,
            embed_dim_noise=0,
            embed_dim_pos=0,
        )
    )


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
                    context_config=config.context,
                    num_groups=config.num_groups,
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
                # Small random init (0.1x default scale) rather than zero.
                # Gives immediate gradient signal for learning Coriolis-like
                # and vorticity-advection couplings, without the variance
                # issues of full-scale random init.
                block.sv_product.weight.data.mul_(0.1)
            if block.pointwise_vv is not None:
                nn.init.zeros_(block.pointwise_vv.weight)
            # Scale down W_sv and initialize W_vv as a mild diagonal
            # contraction. Together these bound vector variance through
            # deep stacks while preserving most of the input signal:
            #   v_new = (1-α)*v_old + scaled_W_sv_contribution
            # With α=0.1 the input retains ~28% through 12 blocks.
            # W_sv is scaled to 0.7x so that the equilibrium variance
            # stays moderate (~4x) despite the weaker contraction.
            block.conv.W_sv.data.mul_(0.7)
            nn.init.zeros_(block.conv.W_vv)
            # Diagonal damping: W_vv[o, o_within_group, 0, 0] = -0.1
            # With groups, output channel o maps to within-group index o % Gv.
            Nv = block.conv.out_channels_vector
            Gv = block.conv.W_vv.shape[1]
            for o in range(Nv):
                block.conv.W_vv.data[o, o % Gv, 0, 0] = -0.1
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
        context: Context | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalars: (B, N_in_s, H, W)
            vectors: (B, N_in_v, H, W, 2)
            context: Optional conditioning context (noise, positional
                embedding, etc.) passed to each block's post-norm.

        Returns:
            scalar_out: (B, N_out_s, H, W)
            vector_out: (B, N_out_v, H, W, 2)
        """
        s = self.scalar_encoder(scalars)
        v = self.vector_encoder(vectors)

        for block in self.blocks:
            s, v = block(s, v, context=context)

        return self.scalar_decoder(s), self.vector_decoder(v)
