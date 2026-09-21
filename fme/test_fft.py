"""Tests for the forked torch-harmonics FFT helpers."""

import pytest
import torch
import torch.fft

from fme.fft import irfft


def _irfft_reference(
    x: torch.Tensor, n: int | None = None, dim: int = -1, **kwargs
) -> torch.Tensor:
    """The original implementation, which zeroes the imaginary parts via
    ``Tensor.imag`` setattr. Kept here to pin down the numerics of ``irfft``.
    """
    if n is None:
        n = 2 * (x.size(dim) - 1)
    x[..., 0].imag = 0.0
    if (n % 2 == 0) and (n // 2 < x.size(dim)):
        x[..., n // 2].imag = 0.0
    return torch.fft.irfft(x, n=n, dim=dim, **kwargs)


def _forward_and_grads(fn, real: torch.Tensor, imag: torch.Tensor, n: int):
    real = real.clone().requires_grad_(True)
    imag = imag.clone().requires_grad_(True)
    y = fn(torch.complex(real, imag), n=n)
    y.pow(2).sum().backward()
    return y, real.grad, imag.grad


@pytest.mark.parametrize("n", [8, 90, 91])
def test_irfft_matches_reference_implementation(n: int):
    """``irfft`` is bit-identical to the reference in output and gradient."""
    torch.manual_seed(0)
    n_freq = n // 2 + 1
    real = torch.randn(3, 4, n_freq, dtype=torch.float32)
    imag = torch.randn(3, 4, n_freq, dtype=torch.float32)

    y_ref, real_grad_ref, imag_grad_ref = _forward_and_grads(
        _irfft_reference, real, imag, n
    )
    y, real_grad, imag_grad = _forward_and_grads(irfft, real, imag, n)

    assert torch.equal(y, y_ref)
    assert torch.equal(real_grad, real_grad_ref)
    assert torch.equal(imag_grad, imag_grad_ref)


def test_irfft_compiles_without_graph_breaks():
    """``irfft`` is traceable by torch.compile, which the SFNO relies on."""
    torch._dynamo.reset()
    try:
        x = torch.randn(3, 4, 5, dtype=torch.complex64)
        explanation = torch._dynamo.explain(lambda x: irfft(x, n=8))(x)
    finally:
        torch._dynamo.reset()
    assert explanation.graph_break_count == 0
