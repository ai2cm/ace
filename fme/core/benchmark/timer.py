import collections
import contextlib
import dataclasses
from typing import Literal, Protocol, Self

import torch


@dataclasses.dataclass
class TimerReport:
    avg_time: float
    children: dict[str, "TimerReport"]


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

    def report(self) -> TimerReport:
        return TimerReport(avg_time=0.0, children={})


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
        return self.start.elapsed_time(self.end)


class CUDATimer:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use CUDATimer.")
        self._children: collections.defaultdict[str, CUDATimer] = (
            collections.defaultdict(CUDATimer)
        )
        self._global_event_pairs: list[EventPair] = []
        self._entered = False

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
        self._global_event_pairs.append(EventPair())
        self._global_event_pairs[-1].record_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._global_event_pairs:
            raise RuntimeError("CUDATimer context was not properly entered.")
        self._global_event_pairs[-1].record_end()
        self._entered = False
        return False

    def child(self, name: str) -> "CUDATimer":
        if not self._entered:
            raise RuntimeError(
                "CUDATimer child cannot be used before entering the timer."
            )
        return self._children[name]

    @property
    def _avg_global_time(self) -> float:
        if len(self._global_event_pairs) == 0:
            raise RuntimeError(
                "CUDATimer report cannot be generated before entering the timer."
            )
        total_time = sum(
            event_pair.elapsed_time_seconds() for event_pair in self._global_event_pairs
        )
        return total_time / len(self._global_event_pairs)

    def _child_reports(self) -> dict[str, TimerReport]:
        return {name: child.report() for name, child in self._children.items()}

    def report(self) -> TimerReport:
        torch.cuda.synchronize()
        return TimerReport(
            avg_time=self._avg_global_time,
            children=self._child_reports(),
        )


__: Timer = CUDATimer()
del __
