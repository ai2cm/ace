import math

import torch

from fme.core.shallow_water.scalar_vector_product import ScalarVectorProduct, VectorDotProduct

H, W = 8, 16


class TestScalarVectorProduct:
    def test_output_shape(self):
        """Output has same shape as input vector."""
        mod = ScalarVectorProduct(3, 2)
        x_s = torch.randn(2, 3, H, W)
        x_v = torch.randn(2, 2, H, W, 2)
        y = mod(x_s, x_v)
        assert y.shape == x_v.shape

    def test_zero_scalar_gives_zero_output(self):
        """Zero scalar input produces zero vector output."""
        mod = ScalarVectorProduct(2, 3)
        x_s = torch.zeros(1, 2, H, W)
        x_v = torch.randn(1, 3, H, W, 2)
        y = mod(x_s, x_v)
        assert y.abs().max() < 1e-10

    def test_zero_vector_gives_zero_output(self):
        """Zero vector input produces zero vector output."""
        mod = ScalarVectorProduct(2, 3)
        x_s = torch.randn(1, 2, H, W)
        x_v = torch.zeros(1, 3, H, W, 2)
        y = mod(x_s, x_v)
        assert y.abs().max() < 1e-10

    def test_pure_scale(self):
        """With only scale weight, output is s * v."""
        mod = ScalarVectorProduct(1, 1)
        with torch.no_grad():
            mod.weight.zero_()
            mod.weight[0, 0, 0] = 1.0  # scale only

        x_s = torch.full((1, 1, H, W), 3.0)
        x_v = torch.randn(1, 1, H, W, 2)
        y = mod(x_s, x_v)
        torch.testing.assert_close(y, 3.0 * x_v)

    def test_pure_rotation(self):
        """With only rotate weight, output is s * (-v, u)."""
        mod = ScalarVectorProduct(1, 1)
        with torch.no_grad():
            mod.weight.zero_()
            mod.weight[0, 0, 1] = 1.0  # rotate only

        x_s = torch.full((1, 1, H, W), 2.0)
        u = torch.ones(1, 1, H, W)
        v = torch.zeros(1, 1, H, W)
        x_v = torch.stack([u, v], dim=-1)

        y = mod(x_s, x_v)
        # 2 * (-0, 1) = (0, 2)
        assert y[..., 0].abs().max() < 1e-6
        torch.testing.assert_close(y[..., 1], torch.full_like(y[..., 1], 2.0))

    def test_coriolis_pattern(self):
        """Reproduces f * (-v, u) with appropriate weights."""
        mod = ScalarVectorProduct(1, 1)
        with torch.no_grad():
            mod.weight.zero_()
            mod.weight[0, 0, 1] = 1.0  # rotate

        # f varies with "latitude" (first spatial dim)
        f = torch.linspace(-1, 1, H).reshape(1, 1, H, 1).expand(1, 1, H, W)
        u = torch.ones(1, 1, H, W) * 3.0
        v = torch.ones(1, 1, H, W) * 2.0
        x_v = torch.stack([u, v], dim=-1)

        y = mod(f, x_v)
        # Expected: f * (-v, u) = f * (-2, 3)
        expected_u = f * (-2.0)
        expected_v = f * 3.0
        torch.testing.assert_close(y[..., 0], expected_u)
        torch.testing.assert_close(y[..., 1], expected_v)

    def test_multiple_scalar_channels_sum(self):
        """Multiple scalar channels contribute additively."""
        mod = ScalarVectorProduct(2, 1)
        with torch.no_grad():
            mod.weight.zero_()
            mod.weight[0, 0, 0] = 1.0  # s0 scales
            mod.weight[1, 0, 1] = 1.0  # s1 rotates

        s0 = torch.full((1, 1, H, W), 2.0)
        s1 = torch.full((1, 1, H, W), 3.0)
        x_s = torch.cat([s0, s1], dim=1)

        u = torch.ones(1, 1, H, W)
        v = torch.zeros(1, 1, H, W)
        x_v = torch.stack([u, v], dim=-1)

        y = mod(x_s, x_v)
        # s0 * scale * (1,0) + s1 * rotate * (1,0)
        # = 2*(1,0) + 3*(0,1) = (2, 3)
        torch.testing.assert_close(y[..., 0], torch.full_like(y[..., 0], 2.0))
        torch.testing.assert_close(y[..., 1], torch.full_like(y[..., 1], 3.0))

    def test_preserves_vector_magnitude_under_pure_rotation(self):
        """Pure rotation preserves |v|."""
        mod = ScalarVectorProduct(1, 1)
        with torch.no_grad():
            mod.weight.zero_()
            mod.weight[0, 0, 1] = 1.0

        x_s = torch.ones(1, 1, H, W)
        x_v = torch.randn(1, 1, H, W, 2)
        y = mod(x_s, x_v)

        mag_in = (x_v[..., 0] ** 2 + x_v[..., 1] ** 2).sqrt()
        mag_out = (y[..., 0] ** 2 + y[..., 1] ** 2).sqrt()
        torch.testing.assert_close(mag_out, mag_in)

    def test_gradient_flows(self):
        """Gradients flow to both inputs and weights."""
        mod = ScalarVectorProduct(2, 2)
        x_s = torch.randn(1, 2, H, W, requires_grad=True)
        x_v = torch.randn(1, 2, H, W, 2, requires_grad=True)
        y = mod(x_s, x_v)
        y.sum().backward()

        assert x_s.grad is not None and x_s.grad.abs().max() > 0
        assert x_v.grad is not None and x_v.grad.abs().max() > 0
        assert mod.weight.grad is not None
        assert mod.weight.grad.abs().max() > 0


class TestVectorDotProduct:
    def test_output_shape(self):
        """Output shape is (B, N_s, H, W)."""
        mod = VectorDotProduct(n_scalar=3, n_vector=2)
        x_v = torch.randn(2, 2, H, W, 2)
        y = mod(x_v)
        assert y.shape == (2, 3, H, W)

    def test_zero_input_gives_zero_output(self):
        """Zero vector input produces zero scalar output."""
        mod = VectorDotProduct(n_scalar=2, n_vector=3)
        x_v = torch.zeros(1, 3, H, W, 2)
        y = mod(x_v)
        assert y.abs().max() < 1e-10

    def test_diagonal_weight_gives_ke(self):
        """Diagonal weight 0.5 computes KE = ½(u² + v²) per channel."""
        K = 3
        mod = VectorDotProduct(n_scalar=K, n_vector=K)
        with torch.no_grad():
            mod.weight.zero_()
            for k in range(K):
                mod.weight[k, k] = 0.5

        x_v = torch.randn(2, K, H, W, 2)
        y = mod(x_v)

        expected = 0.5 * (x_v[..., 0] ** 2 + x_v[..., 1] ** 2)  # (B, K, H, W)
        torch.testing.assert_close(y, expected)

    def test_rotation_invariant(self):
        """Rotating the input vector field leaves the output unchanged."""
        mod = VectorDotProduct(n_scalar=2, n_vector=2)
        x_v = torch.randn(1, 2, H, W, 2)

        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        x_v_rot = torch.stack(
            [c * x_v[..., 0] - s * x_v[..., 1],
             s * x_v[..., 0] + c * x_v[..., 1]],
            dim=-1,
        )

        y     = mod(x_v)
        y_rot = mod(x_v_rot)
        torch.testing.assert_close(y, y_rot)

    def test_linearity_in_weight(self):
        """Doubling the weight doubles the output."""
        mod = VectorDotProduct(n_scalar=2, n_vector=2)
        x_v = torch.randn(1, 2, H, W, 2)

        y1 = mod(x_v)
        with torch.no_grad():
            mod.weight.mul_(2.0)
        y2 = mod(x_v)

        torch.testing.assert_close(y2, 2.0 * y1)

    def test_gradient_flows(self):
        """Gradients flow to the input and weights."""
        mod = VectorDotProduct(n_scalar=2, n_vector=2)
        x_v = torch.randn(1, 2, H, W, 2, requires_grad=True)
        y = mod(x_v)
        y.sum().backward()

        assert x_v.grad is not None and x_v.grad.abs().max() > 0
        assert mod.weight.grad is not None and mod.weight.grad.abs().max() > 0
