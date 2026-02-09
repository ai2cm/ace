import torch

from fme.core.benchmark.benchmark import run_benchmark


def test_run_benchmark():
    def benchmark_fn(timer):
        torch.cuda._sleep(100_000)

    first_result = run_benchmark(benchmark_fn, iters=5, warmup=1)
    assert first_result.timer.total_runs == 5
    second_result = run_benchmark(benchmark_fn, iters=10, warmup=1)
    assert second_result.timer.total_runs == 10
    torch.testing.assert_close(
        first_result.timer.avg_time, second_result.timer.avg_time, rtol=0.05, atol=0
    )
