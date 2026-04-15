"""VectorDiscoBlock: the repeating unit of a scalar-vector network."""

import math

import torch
import torch.nn as nn

from fme.core.disco._vector_convolution import VectorDiscoConvS2, VectorFilterBasis
from fme.core.models.conditional_sfno.layers import (
    ConditionalLayerNorm,
    Context,
    ContextConfig,
)
from fme.core.shallow_water.scalar_vector_product import ScalarVectorProduct


class PointwiseVectorTransform(nn.Module):
    """Pointwise per-channel scale and rotation of vector features.

    Like a 1x1 conv but for vector channels: mixes vector channels
    with learned scale + 90-degree rotation weights.
    Weight shape: (n_out, n_in, 2) where index 0 = scale, 1 = rotate.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        scale = 1.0 / math.sqrt(max(1, n_in))
        self.weight = nn.Parameter(scale * torch.randn(n_out, n_in, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform vector features pointwise.

        Args:
            x: (B, N_in, H, W, 2)

        Returns:
            (B, N_out, H, W, 2)
        """
        u = x[..., 0]  # (B, N_in, H, W)
        v = x[..., 1]
        # w: (N_out, N_in, 2) → (1, N_out, N_in, 1, 1)
        w_s = self.weight[:, :, 0].reshape(1, self.n_out, self.n_in, 1, 1)
        w_r = self.weight[:, :, 1].reshape(1, self.n_out, self.n_in, 1, 1)
        u_in = u.unsqueeze(1)  # (B, 1, N_in, H, W)
        v_in = v.unsqueeze(1)
        out_u = (w_s * u_in + w_r * (-v_in)).sum(dim=2)
        out_v = (w_s * v_in + w_r * u_in).sum(dim=2)
        return torch.stack([out_u, out_v], dim=-1)


class VectorDiscoBlock(nn.Module):
    """One block of the scalar-vector network.

    Applies in sequence:
      1. VectorDiscoConvS2 + pointwise skip — spatial mixing plus
         unfiltered (delta function) scalar and vector paths
      2. Scalar activation — nonlinearity on scalar channels only
      3. Scalar MLP — residual 1x1 conv for scalar channel mixing
      4. ScalarVectorProduct — scalar-dependent vector modulation
      5. Residual connection — add block output to input

    The pointwise skip connections let scalars and vectors pass to
    later stages without spatial smoothing. This is important for
    features like the Coriolis parameter that should not be filtered.

    Input and output channel counts match (required for the residual).
    """

    def __init__(
        self,
        n_scalar: int,
        n_vector: int,
        shape: tuple[int, int],
        vector_filter_basis: VectorFilterBasis | None = None,
        kernel_shape: int | tuple[int, ...] | None = None,
        basis_type: str = "piecewise linear",
        theta_cutoff: float | None = None,
        mlp_hidden_factor: int = 2,
        activation: str = "gelu",
        residual: bool = True,
        post_norm: bool = True,
        context_config: ContextConfig | None = None,
        num_groups: int = 1,
    ):
        """Initialize the block.

        Args:
            n_scalar: number of scalar channels (in and out).
            n_vector: number of vector channels (in and out).
            shape: (nlat, nlon) grid dimensions.
            vector_filter_basis: filter basis for the convolution.
            kernel_shape: convenience alternative to vector_filter_basis.
            basis_type: filter basis type (used with kernel_shape).
            theta_cutoff: filter support radius.
            mlp_hidden_factor: MLP hidden dim = n_scalar * this factor.
            activation: activation function name ("relu", "gelu", "none").
            residual: whether to add the block input to the output.
            post_norm: whether to apply ConditionalLayerNorm to scalars
                after the residual. Set to False for frozen-weight physics
                blocks.
            context_config: conditioning dimensions for the post-norm.
                Required when post_norm is True. When all embed dims are 0,
                the norm is unconditional (plain channel layer norm).
            num_groups: number of channel groups for the convolution,
                pointwise skip, and sv_product. Channels are divided into
                groups that only interact within themselves. The scalar MLP
                and post-norm remain ungrouped to provide cross-group mixing.
        """
        super().__init__()

        self.conv = VectorDiscoConvS2(
            in_channels_scalar=n_scalar,
            in_channels_vector=n_vector,
            out_channels_scalar=n_scalar,
            out_channels_vector=n_vector,
            in_shape=shape,
            out_shape=shape,
            vector_filter_basis=vector_filter_basis,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            theta_cutoff=theta_cutoff,
            bias=True,
            num_groups=num_groups,
        )

        # Pointwise (delta function) skip connections (grouped)
        if n_scalar > 0:
            self.pointwise_ss = nn.Conv2d(
                n_scalar, n_scalar, 1, bias=False, groups=num_groups
            )
        else:
            self.pointwise_ss = None

        if n_vector > 0:
            self.pointwise_vv: PointwiseVectorTransform | None = (
                PointwiseVectorTransform(n_vector, n_vector)
            )
        else:
            self.pointwise_vv = None

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "none": nn.Identity,
        }
        act_cls = activations[activation]

        if n_scalar > 0 and mlp_hidden_factor > 0:
            mlp_hidden = n_scalar * mlp_hidden_factor
            self.scalar_mlp = nn.Sequential(
                nn.Conv2d(n_scalar, mlp_hidden, 1),
                act_cls(),
                nn.Conv2d(mlp_hidden, n_scalar, 1),
            )
        else:
            self.scalar_mlp = None

        if n_scalar > 0 and n_vector > 0:
            self.sv_product: ScalarVectorProduct | None = ScalarVectorProduct(
                n_scalar, n_vector, num_groups=num_groups
            )
        else:
            self.sv_product = None

        self.residual = residual
        self.activation = act_cls()

        # Post-norm: ConditionalLayerNorm on scalars after the residual
        # connection. Only scalars are normalized — vectors pass through
        # unnormalized so their magnitudes remain physically meaningful.
        # Post-norm controls total scalar variance by normalizing the
        # accumulated state (initial + residual), which also controls
        # vector variance through the W_sv path in subsequent blocks.
        # When conditioning dims are all 0, this is a plain channel norm.
        if post_norm and n_scalar > 0:
            if context_config is None:
                context_config = ContextConfig(
                    embed_dim_scalar=0,
                    embed_dim_labels=0,
                    embed_dim_noise=0,
                    embed_dim_pos=0,
                )
            self.scalar_post_norm: ConditionalLayerNorm | None = ConditionalLayerNorm(
                n_channels=n_scalar,
                img_shape=shape,
                context_config=context_config,
            )
        else:
            self.scalar_post_norm = None

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_vector: torch.Tensor,
        context: Context | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the block.

        Args:
            x_scalar: (B, N_s, H, W)
            x_vector: (B, N_v, H, W, 2)
            context: Optional conditioning context for the post-norm.

        Returns:
            y_scalar: (B, N_s, H, W)
            y_vector: (B, N_v, H, W, 2)
        """
        # 1. Convolution + pointwise skip
        s, v = self.conv(x_scalar, x_vector)
        if self.pointwise_ss is not None:
            s = s + self.pointwise_ss(x_scalar)
        if self.pointwise_vv is not None:
            v = v + self.pointwise_vv(x_vector)

        # 2. Scalar activation
        s = self.activation(s)

        # 3. Scalar MLP (residual so zero weights = identity)
        if self.scalar_mlp is not None:
            s = s + self.scalar_mlp(s)

        # 4. ScalarVectorProduct acts on INPUT vectors, not conv output.
        #    This ensures exact Coriolis no-work (f × V_input = 0 power)
        #    while the conv vector output passes through separately.
        if self.sv_product is not None:
            v = v + self.sv_product(s, x_vector)

        # 5. Residual
        if self.residual:
            s, v = x_scalar + s, x_vector + v

        # 6. Scalar post-norm (conditional on context if provided)
        if self.scalar_post_norm is not None:
            if context is None:
                context = Context(
                    embedding_scalar=None,
                    embedding_pos=None,
                    labels=None,
                    noise=None,
                )
            s = self.scalar_post_norm(s, context)

        return s, v
