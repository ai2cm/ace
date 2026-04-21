import pytest
import torch
import torch_harmonics as th

from fme.core.disco import DiscreteContinuousConvS2

# Basis types and grids used in the codebase.
BASIS_TYPES = ["morlet", "piecewise linear"]
GRIDS = ["equiangular", "legendre-gauss"]


@pytest.mark.parametrize("basis_type", BASIS_TYPES + ["isotropic morlet"])
@pytest.mark.parametrize("grid", GRIDS)
def test_forward_shape(basis_type, grid):
    """Basic smoke test: output has the expected shape."""
    img_shape = (16, 32)
    in_channels = 4
    out_channels = 8
    conv = DiscreteContinuousConvS2(
        in_channels,
        out_channels,
        in_shape=img_shape,
        out_shape=img_shape,
        kernel_shape=(3, 3),
        basis_type=basis_type,
        basis_norm_mode="mean",
        grid_in=grid,
        grid_out=grid,
        bias=True,
        theta_cutoff=0.5,
    )
    x = torch.randn(2, in_channels, *img_shape)
    with torch.no_grad():
        y = conv(x)
    assert y.shape == (2, out_channels, *img_shape)


@pytest.mark.parametrize("basis_type", BASIS_TYPES)
@pytest.mark.parametrize("grid", GRIDS)
def test_matches_torch_harmonics_reference(basis_type, grid):
    """Output matches the torch-harmonics sparse-matrix implementation."""
    torch.manual_seed(0)
    img_shape = (16, 32)
    in_channels = 4
    out_channels = 4
    conv_kwargs = dict(
        in_shape=img_shape,
        out_shape=img_shape,
        kernel_shape=(3, 3),
        basis_type=basis_type,
        basis_norm_mode="mean",
        groups=1,
        grid_in=grid,
        grid_out=grid,
        bias=False,
        theta_cutoff=0.5,
    )

    fft_conv = DiscreteContinuousConvS2(
        in_channels,
        out_channels,
        **conv_kwargs,  # type: ignore[arg-type]
    )
    ref_conv = th.DiscreteContinuousConvS2(
        in_channels,
        out_channels,
        **conv_kwargs,  # type: ignore[arg-type]
    )

    # Copy weights from fft_conv to ref_conv so they use the same parameters
    ref_conv.weight.data.copy_(fft_conv.weight.data)

    x = torch.randn(2, in_channels, *img_shape)
    with torch.no_grad():
        y_fft = fft_conv(x)
        y_ref = ref_conv(x)

    torch.testing.assert_close(y_fft, y_ref, atol=1e-4, rtol=1e-4)


def test_grouped_convolution():
    """Test that grouped convolutions work."""
    img_shape = (16, 32)
    in_channels = 8
    out_channels = 8
    groups = 4
    conv = DiscreteContinuousConvS2(
        in_channels,
        out_channels,
        in_shape=img_shape,
        out_shape=img_shape,
        kernel_shape=(3, 3),
        basis_type="morlet",
        basis_norm_mode="mean",
        groups=groups,
        grid_in="equiangular",
        grid_out="equiangular",
        bias=False,
        theta_cutoff=0.5,
    )
    x = torch.randn(2, in_channels, *img_shape)
    with torch.no_grad():
        y = conv(x)
    assert y.shape == (2, out_channels, *img_shape)


def test_backward_pass():
    """Gradients flow through the FFT-based convolution."""
    img_shape = (16, 32)
    in_channels = 4
    out_channels = 4
    conv = DiscreteContinuousConvS2(
        in_channels,
        out_channels,
        in_shape=img_shape,
        out_shape=img_shape,
        kernel_shape=(3, 3),
        basis_type="morlet",
        basis_norm_mode="mean",
        grid_in="equiangular",
        grid_out="equiangular",
        bias=True,
        theta_cutoff=0.5,
    )
    x = torch.randn(2, in_channels, *img_shape)
    y = conv(x)
    y.sum().backward()
    assert conv.weight.grad is not None
    assert conv.bias.grad is not None
