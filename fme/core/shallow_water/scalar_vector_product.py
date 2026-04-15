"""Pointwise scalar-vector interaction modules."""

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


class VectorDotProduct(nn.Module):
    """Pointwise weighted squared-magnitude of vector features → scalar features.

    For each output scalar channel c:

        output[c] = ∑_v  w[c, v] · (u_v² + v_v²)

    The result is frame-invariant (rotation of the input vector field leaves the
    output unchanged), since ``u² + v²`` is the squared Euclidean norm.

    The canonical use-case is kinetic energy: with a diagonal weight ``w[k, k] = 0.5``
    this gives ``KE_k = ½ |V_k|²`` for each level k.

    Weight shape: ``(n_scalar, n_vector)``.
    """

    def __init__(self, n_scalar: int, n_vector: int):
        """
        Args:
            n_scalar: number of output scalar channels.
            n_vector: number of input vector channels.
        """
        super().__init__()
        self.n_scalar = n_scalar
        self.n_vector = n_vector
        scale = 1.0 / math.sqrt(max(1, n_vector))
        self.weight = nn.Parameter(scale * torch.randn(n_scalar, n_vector))

    def forward(self, x_vector: torch.Tensor) -> torch.Tensor:
        """Compute weighted squared magnitudes.

        Args:
            x_vector: ``(B, N_v, H, W, 2)``

        Returns:
            ``(B, N_s, H, W)``
        """
        mag_sq = (x_vector**2).sum(-1)  # (B, N_v, H, W)
        return torch.einsum("sv,bvhw->bshw", self.weight, mag_sq)
