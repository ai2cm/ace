"""Tests for the forked torch-harmonics FFT helpers used by DISCO."""

import pytest
import torch
import torch.fft

from fme.core.disco._fft import irfft


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
    """``irfft`` is traceable by torch.compile, which the DISCO conv relies on."""
    torch._dynamo.reset()
    try:
        x = torch.randn(3, 4, 5, dtype=torch.complex64)
        explanation = torch._dynamo.explain(lambda x: irfft(x, n=8))(x)
    finally:
        torch._dynamo.reset()
    assert explanation.graph_break_count == 0


@pytest.mark.parametrize("n", [8, 91])
def test_irfft_zeroes_dc_and_nyquist_imaginary_parts_in_place(n: int):
    """The zeroing must land on the caller's tensor rather than on a copy.

    The output alone cannot show this: the backend ignores those imaginary
    parts anyway, so the write is only observable on the input.
    """
    torch.manual_seed(0)
    n_freq = n // 2 + 1
    x = torch.complex(torch.randn(3, 4, n_freq), torch.randn(3, 4, n_freq))
    assert (x[..., 0].imag != 0).all(), "inputs should start with nonzero DC imag"
    expected = x.clone()
    expected.imag[..., 0] = 0.0
    if n % 2 == 0:
        expected.imag[..., n // 2] = 0.0
    irfft(x, n=n)
    assert torch.equal(x, expected)


def test_irfft_compiled_matches_eager_with_gradients():
    """A fullgraph compile of ``irfft`` gives bit-identical outputs and grads."""
    torch.manual_seed(0)
    real = torch.randn(3, 4, 5, dtype=torch.float32)
    imag = torch.randn(3, 4, 5, dtype=torch.float32)
    torch._dynamo.reset()
    try:
        # aot_eager runs the full tracing path without inductor's codegen cost
        compiled = torch.compile(irfft, backend="aot_eager", fullgraph=True)
        compiled_results = _forward_and_grads(compiled, real, imag, n=8)
    finally:
        torch._dynamo.reset()
    eager_results = _forward_and_grads(irfft, real, imag, n=8)
    for compiled_result, eager_result in zip(compiled_results, eager_results):
        assert torch.equal(compiled_result, eager_result)
