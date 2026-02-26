import dataclasses

import pytest
import torch

from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.models.conditional_sfno.s2convolutions import SpectralConvS2

from .s2convolutions import _contract_dhconv, _contract_lora, _contract_lora_lowmem


@dataclasses.dataclass
class BenchmarkResult:
    ms_total: float
    ms_per: float
    max_alloc: int
    max_reserved: int
    y_shape: tuple
    y_dtype: torch.dtype


def benchmark(fn, iters=10, warmup=1) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    for _ in range(iters):
        y = fn()
    ender.record()
    torch.cuda.synchronize()

    ms = starter.elapsed_time(ender)
    return BenchmarkResult(
        ms_total=ms,
        ms_per=ms / iters,
        max_alloc=torch.cuda.max_memory_allocated(),
        max_reserved=torch.cuda.max_memory_reserved(),
        y_shape=tuple(y.shape),
        y_dtype=y.dtype,
    )


@pytest.mark.skipif(
    get_device().type != "cuda",
    reason=(
        "This test is only relevant for CUDA since "
        "it's testing speed of DHConv groups on GPU."
    ),
)  # noqa: E501
def test_contract_dhconv_groups_are_faster():
    B = 2
    C = 512
    H = 180
    L = 360
    G = 8
    x = torch.randn(B, 1, C, H, L, dtype=torch.complex64, device=get_device())
    w = torch.randn(1, C, C, H, 2, dtype=torch.float32, device=get_device())

    def contract_ungrouped():
        return _contract_dhconv(x, w)

    ungrouped_result = benchmark(contract_ungrouped)

    x_grouped = x.reshape(B, G, C // G, H, L)
    w_grouped = torch.randn(
        G, C // G, C // G, H, 2, dtype=torch.float32, device=get_device()
    )

    def contract_grouped():
        return _contract_dhconv(x_grouped, w_grouped)

    grouped_result = benchmark(contract_grouped)

    assert grouped_result.ms_per < 2 / G * ungrouped_result.ms_per, (
        "Expected grouped DHConv to be faster than ungrouped, but got "
        f"{grouped_result.ms_per:.6f} seconds for grouped and "
        f"{ungrouped_result.ms_per:.6f} seconds for ungrouped."
    )
    assert grouped_result.max_alloc < ungrouped_result.max_alloc, (
        "Expected grouped DHConv to use less memory than ungrouped, but got "
        f"{grouped_result.max_alloc/1024/1024:.2f} MB for grouped and "
        f"{ungrouped_result.max_alloc/1024/1024:.2f} MB for ungrouped."
    )


@pytest.mark.skipif(
    get_device().type != "cuda",
    reason=(
        "This test is only relevant for CUDA since "
        "it's testing speed of DHConv groups on GPU."
    ),
)  # noqa: E501
def test_lora_lowmem_is_faster():
    B = 2
    C = 512
    R = 32
    H = 180
    L = 360
    x = torch.randn(B, 1, C, H, L, dtype=torch.complex64, device=get_device())
    lora_A = torch.randn(1, C, R, H, 2, dtype=torch.float32, device=get_device())
    lora_B = torch.randn(1, R, C, H, 2, dtype=torch.float32, device=get_device())

    def contract():
        return _contract_lora(lora_A, lora_B, x)

    baseline_result = benchmark(contract)

    def contract_lowmem():
        return _contract_lora_lowmem(lora_A, lora_B, x)

    lowmem_result = benchmark(contract_lowmem)

    theoretical_ratio = (R / C) ** 0.5

    assert theoretical_ratio < 0.5

    assert lowmem_result.ms_per < 2 * theoretical_ratio * baseline_result.ms_per, (
        "Expected LoRA low-memory contraction to be faster than standard, but got "
        f"{lowmem_result.ms_per:.6f} seconds for low-memory and "
        f"{baseline_result.ms_per:.6f} seconds for standard."
    )
    assert lowmem_result.max_alloc < 0.75 * baseline_result.max_alloc, (
        "Expected LoRA low-memory contraction to use significantly less memory "
        "than standard, but got "
        f"{lowmem_result.max_alloc/1024/1024:.2f} MB for low-memory and "
        f"{baseline_result.max_alloc/1024/1024:.2f} MB for standard."
    )


def test_spectral_conv_s2_lora():
    in_channels = 8
    out_channels = in_channels
    n_lat = 12
    n_lon = 24
    operations = LatLonOperations(
        area_weights=torch.ones(n_lat, n_lon),
        grid="legendre-gauss",
    )
    sht = operations.get_real_sht()
    isht = operations.get_real_isht()

    conv1 = SpectralConvS2(
        forward_transform=sht,
        inverse_transform=isht,
        in_channels=in_channels,
        out_channels=out_channels,
        operator_type="dhconv",
        use_tensorly=False,
    )
    assert conv1.lora_A is None
    assert conv1.lora_B is None
    conv2 = SpectralConvS2(
        forward_transform=sht,
        inverse_transform=isht,
        in_channels=in_channels,
        out_channels=out_channels,
        operator_type="dhconv",
        use_tensorly=False,
        lora_rank=4,
        lora_alpha=8,
    )
    assert conv2.lora_A is not None
    assert conv2.lora_B is not None

    conv2.load_state_dict(conv1.state_dict(), strict=False)
    x = torch.randn(2, in_channels, n_lat, n_lon)
    y1, residual1 = conv1(x)
    y2, residual2 = conv2(x)

    # initial outputs should be identical since LoRA starts at 0
    assert torch.allclose(y1, y2, atol=1e-6)
    assert torch.allclose(residual1, residual2, atol=1e-6)
