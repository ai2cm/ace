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

from fme.core.models.conditional_sfno.layers import (
    ChannelLayerNorm,
    Context,
    ContextConfig,
)
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
        scalar_encoder_layers: Number of hidden layers in the scalar
            encoder/decoder MLP. 0 (default) uses a single Conv2d.
            N>=1 uses N x (Conv2d + activation) + final Conv2d,
            matching the SFNO encoder architecture.
        residual_blocks: Whether VectorDiscoBlocks use internal residuals.
        num_groups: Number of channel groups for the convolution.
        embed_dim_noise: Dimension of noise embedding for conditional
            layer norm. 0 disables noise conditioning.
        noise_type: Type of noise ("gaussian" or "isotropic"). Isotropic
            noise has spatially uniform power spectrum on the sphere.
        embed_dim_pos: Dimension of learned positional embedding for
            conditional layer norm. 0 disables positional conditioning.
    """

    n_scalar_channels: int = 64
    n_vector_channels: int = 16
    n_blocks: int = 4
    kernel_shape: int = 5
    mlp_hidden_factor: int = 2
    activation: str = "gelu"
    scalar_encoder_layers: int = 0
    residual_blocks: bool = True
    num_groups: int = 1
    embed_dim_noise: int = 0
    noise_type: str = "gaussian"
    embed_dim_pos: int = 0


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
        self._embed_dim_noise = config.embed_dim_noise
        self._noise_type = config.noise_type
        self._img_shape = img_shape

        # Set up isotropic noise generator if needed
        if config.embed_dim_noise > 0 and config.noise_type == "isotropic":
            from fme.core.distributed import Distributed

            dist = Distributed.get_instance()
            nlat, nlon = img_shape
            self._isht = dist.get_isht(nlat, nlon, grid="equiangular")
            self._lmax = nlat
            self._mmax = nlon // 2 + 1
        else:
            self._isht = None
            self._lmax = 0
            self._mmax = 0

        context_config = ContextConfig(
            embed_dim_scalar=0,
            embed_dim_labels=0,
            embed_dim_noise=config.embed_dim_noise,
            embed_dim_pos=config.embed_dim_pos,
        )

        # Scalar encoder: either a single Conv2d (0 layers) or an MLP
        # with hidden layers matching the SFNO encoder pattern.
        activations = {"relu": nn.ReLU, "gelu": nn.GELU, "none": nn.Identity}
        act_cls = activations[config.activation]
        n_enc_layers = config.scalar_encoder_layers
        if n_enc_layers == 0:
            self.scalar_encoder = nn.Conv2d(n_in_scalars, Ns, 1)
        else:
            enc_modules: list[nn.Module] = []
            current_dim = n_in_scalars
            for _ in range(n_enc_layers):
                enc_modules.append(nn.Conv2d(current_dim, Ns, 1, bias=True))
                enc_modules.append(act_cls())
                current_dim = Ns
            enc_modules.append(nn.Conv2d(current_dim, Ns, 1, bias=False))
            self.scalar_encoder = nn.Sequential(*enc_modules)
        self.scalar_encoder_norm = ChannelLayerNorm(Ns)
        self.vector_encoder = PointwiseVectorTransform(n_in_vectors, Nv)

        # Learned positional embedding (conditions the post-norm in each block)
        if config.embed_dim_pos > 0:
            self.pos_embed: nn.Parameter | None = nn.Parameter(
                torch.zeros(1, config.embed_dim_pos, *img_shape)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

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
                    context_config=context_config,
                    num_groups=config.num_groups,
                )
                for _ in range(config.n_blocks)
            ]
        )

        # Scalar decoder (mirrors encoder structure)
        if n_enc_layers == 0:
            self.scalar_decoder = nn.Conv2d(Ns, n_out_scalars, 1)
        else:
            dec_modules: list[nn.Module] = []
            current_dim = Ns
            for _ in range(n_enc_layers):
                dec_modules.append(nn.Conv2d(current_dim, Ns, 1, bias=True))
                dec_modules.append(act_cls())
                current_dim = Ns
            dec_modules.append(nn.Conv2d(current_dim, n_out_scalars, 1))
            self.scalar_decoder = nn.Sequential(*dec_modules)
        self.vector_decoder = PointwiseVectorTransform(Nv, n_out_vectors)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for well-behaved forward pass.

        The scalar decoder keeps its default (Kaiming) init so that
        diagnostic variables (output-only, no residual) start with
        nonzero predictions. The vector decoder is zero-initialized
        so that wind predictions start as identity (input + 0 = input)
        under residual prediction.
        """
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalars: (B, N_in_s, H, W)
            vectors: (B, N_in_v, H, W, 2)

        Returns:
            scalar_out: (B, N_out_s, H, W)
            vector_out: (B, N_out_v, H, W, 2)
        """
        B = scalars.shape[0]

        # Build context for conditional layer norm
        if self._embed_dim_noise > 0:
            if self._noise_type == "isotropic" and self._isht is not None:
                from fme.ace.registry.stochastic_sfno import isotropic_noise

                noise = isotropic_noise(
                    (B, self._embed_dim_noise),
                    self._lmax,
                    self._mmax,
                    self._isht,
                    device=scalars.device,
                )
            else:
                noise = torch.randn(
                    B,
                    self._embed_dim_noise,
                    *self._img_shape,
                    device=scalars.device,
                    dtype=scalars.dtype,
                )
        else:
            noise = None

        if self.pos_embed is not None:
            embedding_pos = self.pos_embed.expand(B, -1, -1, -1)
        else:
            embedding_pos = None

        context = Context(
            embedding_scalar=None,
            embedding_pos=embedding_pos,
            labels=None,
            noise=noise,
        )

        s = self.scalar_encoder_norm(self.scalar_encoder(scalars))
        v = self.vector_encoder(vectors)

        for block in self.blocks:
            s, v = block(s, v, context=context)

        return self.scalar_decoder(s), self.vector_decoder(v)
