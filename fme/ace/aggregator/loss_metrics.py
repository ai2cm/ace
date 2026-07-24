import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.loss import ChannelLossInfo


class PerStepLossAggregator:
    """Accumulates per-step loss metrics across batches and produces means.

    Ranks may record different key sets — e.g. sparse ``loss_step_N`` keys
    when the stepper evaluates a per-batch sampled step count, unequal batch
    counts per rank, or a rank that records no batches at all — so the
    distributed reduction must not depend on locally-observed keys.
    """

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
        # Every rank must issue an identical sequence of collective calls, so
        # first agree on the global key universe, then reduce dense sum and
        # count vectors over it. Means are count-weighted: ranks contribute in
        # proportion to how many batches they recorded for each key, and keys
        # with a global count of zero are skipped.
        gathered = dist.gather_object(sorted(self._sums.keys()))
        universe: list[str] | None = None
        if gathered is not None:
            universe = sorted(set().union(*gathered))
        keys: list[str] = dist.scatter_object(universe)
        if len(keys) == 0:
            return {}
        device = get_device()
        sums = torch.zeros(len(keys), device=device)
        counts = torch.zeros(len(keys), device=device)
        for i, key in enumerate(keys):
            if key in self._sums:
                sums[i] = self._sums[key].to(device)
                counts[i] = self._counts[key]
        sums = dist.reduce_sum(sums)
        counts = dist.reduce_sum(counts)
        logs: dict[str, float] = {}
        for key, key_sum, key_count in zip(
            keys, sums.cpu().tolist(), counts.cpu().tolist()
        ):
            if key_count > 0:
                logs[f"{label}/mean/{key}"] = key_sum / key_count
        return logs


class PerChannelLossAggregator:
    """Accumulates per-channel (per-variable) loss values across batches.

    Each recorded value carries the number of active samples that
    contributed to that channel's mean, so masked-out samples never
    dilute the aggregated mean across batches.
    """

    def __init__(self):
        self._weighted_sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def record(self, per_channel_losses: dict[str, ChannelLossInfo]) -> None:
        for var_name, info in per_channel_losses.items():
            ws = self._weighted_sums.get(
                var_name,
                torch.tensor(0.0, device=get_device(), dtype=info.loss.dtype),
            )
            self._weighted_sums[var_name] = ws + info.loss * info.count
            self._counts[var_name] = self._counts.get(var_name, 0) + info.count

    def get_logs(self, label: str) -> dict[str, float]:
        dist = Distributed.get_instance()
        logs: dict[str, float] = {}
        for var_name, ws in self._weighted_sums.items():
            count = self._counts[var_name]
            mean = ws / count if count > 0 else ws
            logs[f"{label}/mean/loss/{var_name}"] = float(
                dist.reduce_mean(mean).cpu().numpy()
            )
        return logs
