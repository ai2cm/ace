"""Tests for VectorDiscoNetwork."""

import math

import torch

from fme.core.models.vector_disco import VectorDiscoNetwork, VectorDiscoNetworkConfig

IMG_SHAPE = (16, 32)
N_IN_SCALARS = 10
N_OUT_SCALARS = 8
N_IN_VECTORS = 4
N_OUT_VECTORS = 4


def _make_network(**overrides):
    config = VectorDiscoNetworkConfig(
        n_scalar_channels=16,
        n_vector_channels=4,
        n_blocks=2,
        kernel_shape=3,
        **overrides,
    )
    return VectorDiscoNetwork(
        config=config,
        n_in_scalars=N_IN_SCALARS,
        n_out_scalars=N_OUT_SCALARS,
        n_in_vectors=N_IN_VECTORS,
        n_out_vectors=N_OUT_VECTORS,
        img_shape=IMG_SHAPE,
    )


class TestVectorDiscoNetwork:
    def test_construction(self):
        net = _make_network()
        assert isinstance(net, torch.nn.Module)

    def test_forward_shapes(self):
        net = _make_network()
        B = 2
        scalars = torch.randn(B, N_IN_SCALARS, *IMG_SHAPE)
        vectors = torch.randn(B, N_IN_VECTORS, *IMG_SHAPE, 2)
        s_out, v_out = net(scalars, vectors)
        assert s_out.shape == (B, N_OUT_SCALARS, *IMG_SHAPE)
        assert v_out.shape == (B, N_OUT_VECTORS, *IMG_SHAPE, 2)

    def test_vector_decoder_zero_init(self):
        """Vector decoder is zero-init (identity under residual prediction)."""
        net = _make_network()
        B = 1
        scalars = torch.randn(B, N_IN_SCALARS, *IMG_SHAPE)
        vectors = torch.randn(B, N_IN_VECTORS, *IMG_SHAPE, 2)
        s_out, v_out = net(scalars, vectors)
        # Vector decoder is zero-init → zero output
        assert v_out.abs().max() == 0.0
        # Scalar decoder has default init → nonzero output
        assert s_out.abs().max() > 0.0

    def test_residual_blocks_false(self):
        """Network with residual_blocks=False runs without error."""
        net = _make_network(residual_blocks=False)
        B = 1
        scalars = torch.randn(B, N_IN_SCALARS, *IMG_SHAPE)
        vectors = torch.randn(B, N_IN_VECTORS, *IMG_SHAPE, 2)
        s_out, v_out = net(scalars, vectors)
        assert s_out.shape == (B, N_OUT_SCALARS, *IMG_SHAPE)
        assert v_out.shape == (B, N_OUT_VECTORS, *IMG_SHAPE, 2)

    def test_mlp_encoder(self):
        """Network with scalar_encoder_layers=1 runs correctly."""
        net = _make_network(scalar_encoder_layers=1)
        B = 1
        scalars = torch.randn(B, N_IN_SCALARS, *IMG_SHAPE)
        vectors = torch.randn(B, N_IN_VECTORS, *IMG_SHAPE, 2)
        s_out, v_out = net(scalars, vectors)
        assert s_out.shape == (B, N_OUT_SCALARS, *IMG_SHAPE)
        assert v_out.shape == (B, N_OUT_VECTORS, *IMG_SHAPE, 2)
        assert v_out.abs().max() == 0.0  # vector decoder still zero-init

    def test_gaussian_input_output_variance(self):
        """With random decoder weights, outputs are well-behaved at 12 blocks.

        Verifies that the post-norm (ChannelLayerNorm on scalars after
        each block's residual) controls variance through deep stacks.
        We re-initialize the decoder to produce nonzero output.
        """
        torch.manual_seed(0)
        config = VectorDiscoNetworkConfig(
            n_scalar_channels=8,
            n_vector_channels=4,
            n_blocks=12,
            kernel_shape=3,
        )
        img_shape = (8, 16)
        n_in_s, n_out_s, n_in_v, n_out_v = 6, 4, 3, 3
        net = VectorDiscoNetwork(config, n_in_s, n_out_s, n_in_v, n_out_v, img_shape)
        # Re-init decoder to produce nonzero output
        torch.nn.init.kaiming_uniform_(net.scalar_decoder.weight)
        n_v_in = net.vector_decoder.n_in
        net.vector_decoder.weight.data.normal_(0, 1.0 / math.sqrt(max(1, n_v_in)))

        B = 2
        scalars = torch.randn(B, n_in_s, *img_shape)
        vectors = torch.randn(B, n_in_v, *img_shape, 2)

        with torch.no_grad():
            s_out, v_out = net(scalars, vectors)

        assert torch.isfinite(s_out).all(), "scalar output contains NaN/Inf"
        assert torch.isfinite(v_out).all(), "vector output contains NaN/Inf"

        s_in_var = scalars.var().item()
        v_in_var = vectors.var().item()
        s_out_var = s_out.var().item()
        v_out_var = v_out.var().item()

        s_ratio = s_out_var / max(s_in_var, 1e-10)
        v_ratio = v_out_var / max(v_in_var, 1e-10)

        if s_ratio > 2.0 or s_ratio < 0.5:
            print(
                f"NOTE: scalar variance ratio = {s_ratio:.2f} "
                f"(in={s_in_var:.4f}, out={s_out_var:.4f})"
            )
        if v_ratio > 2.0 or v_ratio < 0.5:
            print(
                f"NOTE: vector variance ratio = {v_ratio:.2f} "
                f"(in={v_in_var:.4f}, out={v_out_var:.4f})"
            )

        # Post-norm controls scalar variance to ~2x regardless of depth.
        # W_vv damping init makes vector residual updates contractive,
        # bounding vector variance to ~2-3x regardless of depth.
        assert s_ratio < 10.0, f"scalar variance exploded: ratio={s_ratio:.2f}"
        assert v_ratio < 10.0, f"vector variance exploded: ratio={v_ratio:.2f}"
        assert s_ratio > 0.1, f"scalar variance collapsed: ratio={s_ratio:.2f}"
        assert v_ratio > 0.1, f"vector variance collapsed: ratio={v_ratio:.2f}"

    def test_gradient_flows(self):
        """Gradients flow through the full network."""
        net = _make_network()
        # Re-init decoder so output is nonzero
        torch.nn.init.kaiming_uniform_(net.scalar_decoder.weight)
        n_v_in = net.vector_decoder.n_in
        net.vector_decoder.weight.data.normal_(0, 1.0 / math.sqrt(max(1, n_v_in)))

        scalars = torch.randn(1, N_IN_SCALARS, *IMG_SHAPE, requires_grad=True)
        vectors = torch.randn(1, N_IN_VECTORS, *IMG_SHAPE, 2, requires_grad=True)
        s_out, v_out = net(scalars, vectors)
        loss = s_out.sum() + v_out.sum()
        loss.backward()
        assert scalars.grad is not None
        assert vectors.grad is not None
        assert scalars.grad.abs().max() > 0
        assert vectors.grad.abs().max() > 0
