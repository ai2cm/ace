import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed


class PerStepLossAggregator:
    """Accumulates per-step loss metrics across batches and produces means."""

    def __init__(self):
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def record(self, metrics: dict[str, torch.Tensor]) -> None:
        for key, value in metrics.items():
            if key not in self._sums:
                self._sums[key] = value.detach().clone()
                self._counts[key] = 1
            else:
                self._sums[key] = self._sums[key] + value.detach()
                self._counts[key] += 1

    def get_logs(self, label: str) -> dict[str, float]:
        dist = Distributed.get_instance()
        logs: dict[str, float] = {}
        for key in sorted(self._sums.keys()):
            count = self._counts[key]
            logs[f"{label}/mean/{key}"] = float(
                dist.reduce_mean(self._sums[key] / count).cpu().numpy()
            )
        return logs


class PerChannelLossAggregator:
    """Accumulates per-channel (per-variable) loss values across batches."""

    def __init__(self):
        self._sums: dict[str, torch.Tensor] = {}
        self._n_batches = 0

    def record(self, per_channel_losses: dict[str, torch.Tensor]) -> None:
        self._n_batches += 1
        for var_name, value in per_channel_losses.items():
            acc = self._sums.get(
                var_name,
                torch.tensor(0.0, device=get_device(), dtype=value.dtype),
            )
            self._sums[var_name] = acc + value

    def get_logs(self, label: str) -> dict[str, float]:
        dist = Distributed.get_instance()
        logs: dict[str, float] = {}
        if self._n_batches > 0:
            for var_name, acc in self._sums.items():
                logs[f"{label}/mean/loss/{var_name}"] = float(
                    dist.reduce_mean(acc / self._n_batches).cpu().numpy()
                )
        return logs
