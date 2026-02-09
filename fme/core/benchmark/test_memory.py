import pytest
import torch

from fme.core.benchmark.memory import MemoryResult, benchmark_memory


def test_cannot_nest_benchmark():
    with benchmark_memory():
        with pytest.raises(RuntimeError, match="benchmark_memory cannot be nested"):
            with benchmark_memory():
                pass


def test_cannot_get_result_before_end():
    with benchmark_memory() as bm:
        with pytest.raises(RuntimeError, match="MemoryBenchmark is still running"):
            bm.result


def test_larger_array_uses_larger_memory():
    with benchmark_memory() as bm1:
        _ = torch.randn(100, 100, device="cuda")
    with benchmark_memory() as bm2:
        _ = torch.randn(200, 200, device="cuda")

    assert bm2.result.max_alloc > bm1.result.max_alloc


def test_memory_benchmark_combine():
    result1 = MemoryResult(max_alloc=100, max_reserved=200)
    result2 = MemoryResult(max_alloc=150, max_reserved=180)
    combined = result1.combine(result2)
    assert combined.max_alloc == 150
    assert combined.max_reserved == 200
