import collections
import contextlib
import dataclasses
from typing import Protocol, Self

import torch


@dataclasses.dataclass
class TimerReport:
    average_time_seconds: dict[str, float]
    children: dict[str, "TimerReport"]


class Timer(Protocol):
    def context(self, name: str) -> contextlib.AbstractContextManager[None]: ...
    def child(self, name: str) -> Self: ...


class NullTimer:
    def context(self, name: str) -> contextlib.nullcontext:
        return contextlib.nullcontext()

    def child(self, name: str) -> "Self":
        return self


_: Timer = NullTimer()
del _


class CUDATimer:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use CUDATimer.")
        self._starters = []
        self._enders = []
        self._names = []
        self._children: collections.defaultdict[str, CUDATimer] = (
            collections.defaultdict(CUDATimer)
        )

    @classmethod
    def new_if_available(cls) -> "CUDATimer" | NullTimer:
        if torch.cuda.is_available():
            return cls()
        else:
            return NullTimer()

    @contextlib.contextmanager
    def context(self, name: str):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        self._starters.append(starter)
        self._enders.append(ender)
        self._names.append(name)
        stream = torch.cuda.current_stream()
        starter.record(stream)
        try:
            yield
        finally:
            ender.record(stream)
        return

    def child(self, name: str) -> "CUDATimer":
        return self._children[name]

    def report(self) -> TimerReport:
        torch.cuda.synchronize()
        total_time_seconds: dict[str, float] = collections.defaultdict(float)
        counts: dict[str, int] = collections.defaultdict(int)
        for starter, ender, name in zip(self._starters, self._enders, self._names):
            total_time_seconds[name] += starter.elapsed_time(ender)
            counts[name] += 1
        average_time_seconds = {
            name: total / counts[name] for name, total in total_time_seconds.items()
        }
        children = {}
        for name, child in self._children.items():
            children[name] = child.report()
        return TimerReport(average_time_seconds=average_time_seconds, children=children)


__: Timer = CUDATimer()
del __
