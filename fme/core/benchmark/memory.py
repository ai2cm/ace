from typing import Literal, NamedTuple

import torch

_benchmark_memory_started = False


class MemoryBenchmarkResult(NamedTuple):
    max_alloc: int
    max_reserved: int


class MemoryBenchmark:
    def __init__(self):
        self._started = False
        self._ended = False

    def __enter__(self) -> "MemoryBenchmark":
        global _benchmark_memory_started
        if _benchmark_memory_started:
            raise RuntimeError(
                "benchmark_memory cannot be nested due to its use of globals"
            )
        _benchmark_memory_started = True
        if self._started:
            raise RuntimeError(
                "MemoryBenchmark cannot be nested due to its use of globals"
            )
        if self._ended:
            raise RuntimeError("MemoryBenchmark cannot be reused after it has ended.")
        self._started = True
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._max_alloc = 0
        self._max_reserved = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        torch.cuda.synchronize()
        global _benchmark_memory_started
        _benchmark_memory_started = False
        self._started = False
        self._ended = True
        self._max_alloc = torch.cuda.max_memory_allocated()
        self._max_reserved = torch.cuda.max_memory_reserved()
        return False  # Don't suppress exceptions

    @property
    def result(self) -> MemoryBenchmarkResult:
        if self._started:
            raise RuntimeError(
                "MemoryBenchmark is still running. "
                "Please exit the context before getting results."
            )
        if not self._ended:
            raise RuntimeError(
                "MemoryBenchmark has not been run yet. "
                "Please enter and exit the context before getting results."
            )
        return MemoryBenchmarkResult(
            max_alloc=self._max_alloc, max_reserved=self._max_reserved
        )


def benchmark_memory() -> MemoryBenchmark:
    return MemoryBenchmark()
