import dataclasses
from collections.abc import Callable

import torch

from fme.core.benchmark.memory import MemoryResult, benchmark_memory
from fme.core.benchmark.timer import CUDATimer, NullTimer, Timer, TimerResult
from fme.core.typing_ import TensorDict


@dataclasses.dataclass
class BenchmarkResult:
    memory: MemoryResult
    timer: TimerResult

    def __repr__(self) -> str:
        return f"BenchmarkResult(memory={self.memory}, timer={self.timer})"


BenchmarkFn = Callable[[Timer], TensorDict]


_BENCHMARKS: dict[str, BenchmarkFn] = {}


def register_benchmark(name: str) -> Callable[[BenchmarkFn], BenchmarkFn]:
    def _register(fn: BenchmarkFn) -> BenchmarkFn:
        if name in _BENCHMARKS:
            raise ValueError(f"Benchmark with name '{name}' is already registered.")
        _BENCHMARKS[name] = fn
        return fn

    return _register


def get_benchmarks() -> dict[str, BenchmarkFn]:
    return _BENCHMARKS.copy()


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
