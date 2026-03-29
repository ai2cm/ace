"""Pointwise scalar-vector interaction module."""

import math

import torch
import torch.nn as nn


class ScalarVectorProduct(nn.Module):
    """Pointwise scaling and rotation of vector features by scalar features.

    For each (scalar channel, vector channel) pair, applies a learned
    combination of scaling (preserving direction) and 90-degree rotation:

        v_out += w_scale * s * (u, v) + w_rotate * s * (-v, u)

    This is equivalent to pointwise complex multiplication, where the
    scalar field provides the magnitude/angle and the vector field is
    the complex input. The operation is frame-consistent (pointwise,
    same meridian frame at each grid point).
    """

    def __init__(self, n_scalar: int, n_vector: int):
        """
        Args:
            n_scalar: number of input scalar channels.
            n_vector: number of input/output vector channels.
        """
        super().__init__()
        self.n_scalar = n_scalar
        self.n_vector = n_vector
        scale = 1.0 / math.sqrt(max(1, n_scalar))
        self.weight = nn.Parameter(scale * torch.randn(n_scalar, n_vector, 2))

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Apply scalar-vector product.

        Args:
            x_scalar: (B, N_s, H, W)
            x_vector: (B, N_v, H, W, 2)

        Returns:
            Vector output (B, N_v, H, W, 2), same shape as x_vector.
        """
        # s: (B, N_s, H, W) → (B, N_s, 1, H, W)
        s = x_scalar.unsqueeze(2)
        u = x_vector[..., 0].unsqueeze(1)  # (B, 1, N_v, H, W)
        v = x_vector[..., 1].unsqueeze(1)  # (B, 1, N_v, H, W)

        # w: (N_s, N_v) for each of scale and rotate
        w_s = self.weight[:, :, 0].reshape(1, self.n_scalar, self.n_vector, 1, 1)
        w_r = self.weight[:, :, 1].reshape(1, self.n_scalar, self.n_vector, 1, 1)

        # (B, N_s, N_v, H, W) for each component, sum over N_s
        out_u = (s * (w_s * u + w_r * (-v))).sum(dim=1)
        out_v = (s * (w_s * v + w_r * u)).sum(dim=1)
        return torch.stack([out_u, out_v], dim=-1)
