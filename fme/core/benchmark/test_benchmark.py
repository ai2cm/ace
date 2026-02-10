import json
import os

import pytest
import torch

import fme  # to trigger registration of benchmarks
from fme.core.benchmark.benchmark import BenchmarkABC, BenchmarkResult, get_benchmarks
from fme.core.benchmark.run import get_benchmark_label, get_device_name
from fme.core.testing.regression import validate_tensor_dict

del fme

DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_run_benchmark():
    def benchmark_fn(timer):
        torch.cuda._sleep(100_000_000)

    benchmark = BenchmarkABC.new_from_fn(benchmark_fn)

    first_result = benchmark.run_benchmark(iters=15, warmup=1)
    assert first_result.timer.total_runs == 15
    second_result = benchmark.run_benchmark(iters=20, warmup=1)
    assert second_result.timer.total_runs == 20
    torch.testing.assert_close(
        first_result.timer.avg_time, second_result.timer.avg_time, rtol=0.2, atol=0
    )


def test_benchmarks_are_not_empty():
    assert (
        len(get_benchmarks()) > 0
    ), "No benchmarks were registered, but at least one was expected."


BENCHMARKS = get_benchmarks()


def validate_benchmark_result(
    x: BenchmarkResult, filename_root: str, name: str, **assert_close_kwargs
):
    device_name = get_device_name().replace(" ", "_").replace("/", "_").lower()
    json_filename = f"{filename_root}-{device_name}.json"
    if not os.path.exists(json_filename):
        with open(json_filename, "w") as f:
            json.dump(x.asdict(), f, indent=4)
        png_filename = f"{filename_root}-{device_name}.png"
        label = get_benchmark_label(name)
        x.to_png(png_filename, label=label)
        pytest.fail(f"Regression file {json_filename} did not exist, so it was created")
    else:
        with open(json_filename) as f:
            d = json.load(f)
        y = BenchmarkResult.from_dict(d)
        x.assert_close(y, **assert_close_kwargs)


@pytest.mark.parametrize("benchmark_name", BENCHMARKS.keys())
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_regression(benchmark_name: str):
    benchmark_cls = BENCHMARKS[benchmark_name]
    regression_result = benchmark_cls.run_regression()
    if regression_result is None:
        pytest.skip("Benchmark does not have regression targets.")
    # If run_regression returns something, we expect it to be a TensorDict of results
    assert isinstance(regression_result, dict)
    validate_tensor_dict(
        regression_result,
        os.path.join(DIR, "testdata", f"{benchmark_name}-regression.pt"),
    )


@pytest.mark.parametrize("benchmark_name", BENCHMARKS.keys())
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_benchmark(benchmark_name: str):
    benchmark_cls = BENCHMARKS[benchmark_name]
    result = benchmark_cls.run_benchmark(iters=20, warmup=5)
    validate_benchmark_result(
        result,
        os.path.join(DIR, "testdata", f"{benchmark_name}-benchmark"),
        name=benchmark_name,
        rtol=0.02,
        children_rtol=0.05,  # looser tolerance on sub-timers
    )
