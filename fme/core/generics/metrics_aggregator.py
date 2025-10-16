import collections

import torch

from fme.core.distributed import Distributed


class MetricsAggregator:
    def __init__(self):
        self._metrics = {}
        self._counts = collections.defaultdict(int)

    def record(self, metrics: dict[str, float]):
        for name, value in metrics.items():
            self._add_metric(name, value)

    def _add_metric(self, name: str, value: float):
        if name not in self._metrics:
            self._metrics[name] = value
        else:
            self._metrics[name] += value
        self._counts[name] += 1

    def get_metrics(self) -> dict[str, float]:
        dist = Distributed.get_instance()
        with torch.no_grad():
            metrics = {
                name: dist.reduce_mean(metric / self._counts[name])
                for name, metric in sorted(self._metrics.items())
            }
        return metrics

    def clear(self):
        self._metrics.clear()
        self._counts.clear()
