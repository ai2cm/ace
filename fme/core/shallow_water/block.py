"""VectorDiscoBlock: the repeating unit of a scalar-vector network."""

import torch
import torch.nn as nn

from fme.core.disco._vector_convolution import VectorDiscoConvS2, VectorFilterBasis
from fme.core.shallow_water.scalar_vector_product import ScalarVectorProduct


class VectorDiscoBlock(nn.Module):
    """One block of the scalar-vector network.

    Applies in sequence:
      1. VectorDiscoConvS2 — spatial mixing via all four type paths
      2. Scalar activation — nonlinearity on scalar channels only
      3. Scalar MLP — 1x1 conv for scalar channel mixing
      4. ScalarVectorProduct — scalar-dependent vector modulation
      5. Residual connection — add block output to input

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
    ):
        """
        Args:
            n_scalar: number of scalar channels (in and out).
            n_vector: number of vector channels (in and out).
            shape: (nlat, nlon) grid dimensions.
            vector_filter_basis: filter basis for the convolution.
            kernel_shape: convenience alternative to vector_filter_basis.
            basis_type: filter basis type (used with kernel_shape).
            theta_cutoff: filter support radius.
            mlp_hidden_factor: MLP hidden dim = n_scalar * this factor.
            activation: activation function name ("relu", "gelu").
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
        )

        activations = {"relu": nn.ReLU, "gelu": nn.GELU, "none": nn.Identity}
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
                n_scalar, n_vector
            )
        else:
            self.sv_product = None

        self.activation = act_cls()

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the block.

        Args:
            x_scalar: (B, N_s, H, W)
            x_vector: (B, N_v, H, W, 2)

        Returns:
            y_scalar: (B, N_s, H, W)
            y_vector: (B, N_v, H, W, 2)
        """
        # 1. Convolution
        s, v = self.conv(x_scalar, x_vector)

        # 2. Scalar activation
        s = self.activation(s)

        # 3. Scalar MLP (residual so zero weights = identity)
        if self.scalar_mlp is not None:
            s = s + self.scalar_mlp(s)

        # 4. ScalarVectorProduct (residual: conv vectors pass through,
        #    sv_product adds scalar-dependent corrections)
        if self.sv_product is not None:
            v = v + self.sv_product(s, v)

        # 5. Residual
        return x_scalar + s, x_vector + v
