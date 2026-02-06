import collections
import contextlib

import torch


class CUDATimer:
    def __init__(self):
        self._starters = []
        self._enders = []
        self._names = []

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

    def report(self):
        torch.cuda.synchronize()
        total_times: dict[str, float] = collections.defaultdict(float)
        counts: dict[str, int] = collections.defaultdict(int)
        for starter, ender, name in zip(self._starters, self._enders, self._names):
            total_times[name] += starter.elapsed_time(ender)
            counts[name] += 1
        avg_times = {name: total / counts[name] for name, total in total_times.items()}
        return avg_times


class NullTimer:
    def context(self, name: str) -> contextlib.nullcontext:
        return contextlib.nullcontext()
