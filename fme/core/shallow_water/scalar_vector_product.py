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

    def __init__(self, n_scalar: int, n_vector: int, num_groups: int = 1):
        """
        Args:
            n_scalar: number of input scalar channels.
            n_vector: number of input/output vector channels.
            num_groups: number of channel groups. Each group of Gs=n_scalar/G
                scalars interacts only with its Gv=n_vector/G vectors.
        """
        super().__init__()
        self.n_scalar = n_scalar
        self.n_vector = n_vector
        self.num_groups = num_groups
        Gs = n_scalar // num_groups
        scale = 1.0 / math.sqrt(max(1, Gs))
        # Weight shape: (n_scalar, Gv, 2) — each scalar channel interacts
        # only with the Gv vector channels in its group.
        Gv = n_vector // num_groups
        self.weight = nn.Parameter(scale * torch.randn(n_scalar, Gv, 2))

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
        B, N_s, H, W = x_scalar.shape
        G = self.num_groups
        Gs = N_s // G
        Gv = self.weight.shape[1]

        # Reshape to (B, G, Gs, H, W) and (B, G, Gv, H, W, 2)
        s = x_scalar.reshape(B, G, Gs, H, W).unsqueeze(3)  # (B, G, Gs, 1, H, W)
        u = x_vector[..., 0].reshape(B, G, Gv, H, W).unsqueeze(2)  # (B,G,1,Gv,H,W)
        v = x_vector[..., 1].reshape(B, G, Gv, H, W).unsqueeze(2)

        # w: (G, Gs, Gv, 2)
        w = self.weight.reshape(G, Gs, Gv, 2)
        w_s = w[:, :, :, 0].reshape(1, G, Gs, Gv, 1, 1)
        w_r = w[:, :, :, 1].reshape(1, G, Gs, Gv, 1, 1)

        # (B, G, Gs, Gv, H, W) for each component, sum over Gs
        out_u = (s * (w_s * u + w_r * (-v))).sum(dim=2)  # (B, G, Gv, H, W)
        out_v = (s * (w_s * v + w_r * u)).sum(dim=2)
        return torch.stack([out_u, out_v], dim=-1).reshape(B, -1, H, W, 2)


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
