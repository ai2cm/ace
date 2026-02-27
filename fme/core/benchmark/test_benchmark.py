import os

import pytest
import torch

import fme  # to trigger registration of benchmarks
from fme.core.benchmark.benchmark import BenchmarkABC, get_benchmarks
from fme.core.device import force_cpu
from fme.core.rand import set_seed
from fme.core.testing.regression import validate_tensor_dict

del fme

DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_run_benchmark():
    def benchmark_fn(timer):
        torch.cuda._sleep(100_000_000)

    benchmark = BenchmarkABC.new_from_fn(benchmark_fn)

    first_result = benchmark.run_benchmark(iters=15, warmup=1)
    assert first_result.timer.count == 15
    second_result = benchmark.run_benchmark(iters=20, warmup=1)
    assert second_result.timer.count == 20
    torch.testing.assert_close(
        first_result.timer.avg_time, second_result.timer.avg_time, rtol=0.2, atol=0
    )


def test_benchmarks_are_not_empty():
    assert (
        len(get_benchmarks()) > 0
    ), "No benchmarks were registered, but at least one was expected."


BENCHMARKS = get_benchmarks()


@pytest.mark.parametrize("benchmark_name", BENCHMARKS.keys())
def test_regression(benchmark_name: str):
    with force_cpu():
        set_seed(0)
        benchmark_cls = BENCHMARKS[benchmark_name]
        regression_result = benchmark_cls.run_regression()
        if regression_result is None:
            pytest.skip("Benchmark does not have regression targets.")
        # If run_regression returns something,
        # we expect it to be a TensorDict of results
        assert isinstance(regression_result, dict)
        validate_tensor_dict(
            regression_result,
            os.path.join(DIR, "testdata", f"{benchmark_name}-regression.pt"),
        )
