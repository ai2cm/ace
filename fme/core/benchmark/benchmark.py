import abc
import dataclasses
from collections.abc import Callable
from typing import Self, TypeVar

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


T = TypeVar("T")


class BenchmarkABC(abc.ABC):
    @classmethod
    def new_from_fn(
        cls,
        fn: Callable[[Timer], TensorDict],
    ) -> "BenchmarkABC":
        class FnBenchmark(BenchmarkABC):
            @classmethod
            def new(cls) -> "FnBenchmark":
                return FnBenchmark()

            def run_instance(self, timer: Timer) -> TensorDict:
                return fn(timer)

        return FnBenchmark()

    @classmethod
    @abc.abstractmethod
    def new(cls: type[Self]) -> Self:
        """
        Initialize any state needed for the benchmark.
        This will be called once before the benchmark is run.
        """
        pass

    @classmethod
    def new_for_regression(cls: type[Self]) -> Self | None:
        """
        Initialize any state needed for regression testing.
        This will be called once before regression tests are run.

        If regression testing is not needed, this can return None,
        and regression testing will not be run.

        This exists as a separate method from new so that it can
        use small data sizes more conducive to storing regression targets in git.
        """
        return None

    @classmethod
    def run_benchmark(cls, iters=10, warmup=1) -> BenchmarkResult:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run benchmark.")
        null_timer = NullTimer()
        benchmark = cls.new()
        for _ in range(warmup):
            benchmark.run_instance(null_timer)
        timer = CUDATimer()
        with benchmark_memory() as bm:
            for _ in range(iters):
                with timer:
                    benchmark.run_instance(timer)
        return BenchmarkResult(
            timer=timer.result,
            memory=bm.result,
        )

    @classmethod
    def run_regression(cls) -> TensorDict | None:
        benchmark = cls.new_for_regression()
        if benchmark is None:
            return None
        null_timer = NullTimer()
        return benchmark.run_instance(null_timer)

    @abc.abstractmethod
    def run_instance(self: Self, timer: Timer) -> TensorDict:
        """
        Run the benchmark. This will be called multiple times,
        and should return a TensorDict of results.

        This must not mutate any state on self, since the same instance may be
        used across multiple iterations.
        """
        pass


_BENCHMARKS: dict[str, type[BenchmarkABC]] = {}


def register_benchmark(name: str) -> Callable[[type[BenchmarkABC]], type[BenchmarkABC]]:
    def _register(fn: type[BenchmarkABC]) -> type[BenchmarkABC]:
        if name in _BENCHMARKS:
            raise ValueError(f"Benchmark with name '{name}' is already registered.")
        _BENCHMARKS[name] = fn
        return fn

    return _register


def get_benchmarks() -> dict[str, type[BenchmarkABC]]:
    return _BENCHMARKS.copy()
