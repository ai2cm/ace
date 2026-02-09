import collections
import contextlib
import dataclasses
from typing import Literal, Protocol, Self

import torch


@dataclasses.dataclass
class TimerResult:
    total_runs: int
    avg_time: float
    children: dict[str, "TimerResult"]


class Timer(Protocol):
    def child(self, name: str) -> Self: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]: ...


class NullTimer:
    def context(self, name: str) -> contextlib.nullcontext:
        return contextlib.nullcontext()

    def child(self, name: str) -> "Self":
        return self

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        return False

    def report(self) -> TimerResult:
        return TimerResult(total_runs=0, avg_time=0.0, children={})


_: Timer = NullTimer()
del _


class EventPair:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self._stream = None
        self._start_recorded = False
        self._end_recorded = False

    def record_start(self):
        if self._start_recorded:
            raise RuntimeError(
                "record_start has already been called on this EventPair."
            )
        self._stream = torch.cuda.current_stream()
        self.start.record(self._stream)
        self._start_recorded = True

    def record_end(self):
        if not self._start_recorded:
            raise RuntimeError("record_start must be called before record_end")
        if self._end_recorded:
            raise RuntimeError("record_end has already been called on this EventPair.")
        if self._stream is None:
            raise RuntimeError("record_start must be called before record_end")
        self.end.record(self._stream)
        self._end_recorded = True

    def elapsed_time_seconds(self) -> float:
        if not self._start_recorded or not self._end_recorded:
            raise RuntimeError(
                "Both record_start and record_end must be called "
                "before elapsed_time_seconds can be called."
            )
        return self.start.elapsed_time(self.end) / 1000.0


class CUDATimer:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use CUDATimer.")
        self._children: collections.defaultdict[str, CUDATimer] = (
            collections.defaultdict(CUDATimer)
        )
        self._event_pairs: list[EventPair] = []
        self._entered = False
        self._result: TimerResult | None = None

    @classmethod
    def new_if_available(cls) -> "CUDATimer | NullTimer":
        if torch.cuda.is_available():
            return cls()
        else:
            return NullTimer()

    def __enter__(self):
        if self._entered:
            raise RuntimeError("CUDATimer is already entered.")
        self._entered = True
        self._event_pairs.append(EventPair())
        self._event_pairs[-1].record_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._event_pairs:
            raise RuntimeError("CUDATimer context was not properly entered.")
        self._event_pairs[-1].record_end()
        self._entered = False
        return False

    def child(self, name: str) -> "CUDATimer":
        if not self._entered:
            raise RuntimeError(
                "CUDATimer child cannot be used before entering the timer."
            )
        return self._children[name]

    @property
    def _avg_time(self) -> float:
        if len(self._event_pairs) == 0:
            raise RuntimeError(
                "CUDATimer report cannot be generated before entering the timer."
            )
        total_time = sum(
            event_pair.elapsed_time_seconds() for event_pair in self._event_pairs
        )
        return total_time / len(self._event_pairs)

    def _child_reports(self) -> dict[str, TimerResult]:
        return {name: child.result for name, child in self._children.items()}

    @property
    def result(self) -> TimerResult:
        if self._result is None:
            torch.cuda.synchronize()
            self._result = TimerResult(
                total_runs=len(self._event_pairs),
                avg_time=self._avg_time,
                children=self._child_reports(),
            )
        return self._result


__: Timer = CUDATimer()
del __
