import pytest
import torch

from fme.core.benchmark import memory
from fme.core.benchmark.memory import MemoryResult, benchmark_memory


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cannot_nest_benchmark():
    with benchmark_memory():
        with pytest.raises(RuntimeError, match="benchmark_memory cannot be nested"):
            with benchmark_memory():
                pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cannot_get_result_before_end():
    with benchmark_memory() as bm:
        with pytest.raises(RuntimeError, match="MemoryBenchmark is still running"):
            bm.result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_ensure_cannot_restart():
    assert (
        not memory._benchmark_memory_started
    ), "Global state should be reset before test"
    with benchmark_memory() as bm:
        pass
    with pytest.raises(RuntimeError, match="cannot be reused after it has ended"):
        bm.__enter__()  # Attempt to restart after it has ended
    assert (
        not memory._benchmark_memory_started
    ), "Global state should be reset after test"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_larger_array_uses_larger_memory():
    with benchmark_memory() as bm1:
        _ = torch.randn(100, 100, device="cuda")
    with benchmark_memory() as bm2:
        _ = torch.randn(200, 200, device="cuda")

    assert bm2.result.max_alloc > bm1.result.max_alloc


@pytest.mark.parametrize(
    "v1, v2, rtol, expect_raise",
    [
        (100, 101, 0.02, False),  # within 2%
        (100, 103, 0.02, True),  # outside 2%
        (100, 102, 0.02, False),  # exactly 2% is considered inside
        (10000, 10201, 0.02, True),  # more than 2% is considered outside
        (10201, 10000, 0.02, False),  # it's 2% of first value, so this is inside
        (100, 102, 0.03, False),  # exactly 2% is within 3%
    ],
)
@pytest.mark.parametrize("reserved", [False, True])
def test_assert_close(
    v1: int, v2: int, rtol: float, reserved: bool, expect_raise: bool
):
    if reserved:
        result1 = MemoryResult(max_alloc=0, max_reserved=v1)
        result2 = MemoryResult(max_alloc=0, max_reserved=v2)
    else:
        result1 = MemoryResult(max_alloc=v1, max_reserved=0)
        result2 = MemoryResult(max_alloc=v2, max_reserved=0)
    if expect_raise:
        with pytest.raises(AssertionError):
            result2.assert_close(result1, rtol=rtol)
    else:
        result2.assert_close(result1, rtol=rtol)
