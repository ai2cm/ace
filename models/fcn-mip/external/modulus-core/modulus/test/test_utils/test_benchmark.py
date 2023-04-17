import pytest
import torch
from modulus.utils.benchmark import timeit


skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="There is no GPU to run this test"
)


@skip_if_no_gpu
def test_timeit():
    def func():
        torch.zeros(2**20, device="cuda").exp().cos().sin()

    cpu_timing_ms = timeit(func, cpu_timing=False)
    cuda_event_timing_ms = timeit(func, cpu_timing=True)
    assert cpu_timing_ms - cuda_event_timing_ms < 0.1


if __name__ == "__main__":
    test_timeit()
