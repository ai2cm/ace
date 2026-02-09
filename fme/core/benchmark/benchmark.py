import dataclasses
from collections.abc import Callable

import torch

from fme.core.benchmark.memory import MemoryResult, benchmark_memory
from fme.core.benchmark.timer import CUDATimer, NullTimer, Timer, TimerResult


@dataclasses.dataclass
class BenchmarkResult:
    memory: MemoryResult
    timer: TimerResult


BenchmarkFn = Callable[[Timer], None]


def run_benchmark(fn: BenchmarkFn, iters=10, warmup=1) -> BenchmarkResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run benchmark.")
    null_timer = NullTimer()
    for _ in range(warmup):
        fn(null_timer)
    timer = CUDATimer()
    with benchmark_memory() as bm:
        for _ in range(iters):
            with timer:
                fn(timer)
    return BenchmarkResult(
        timer=timer.result,
        memory=bm.result,
    )
