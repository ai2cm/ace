from typing import Mapping

import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed


class TrainAggregator:
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(self):
        self._n_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        i_time_start: int = 0,
    ):
        if i_time_start != 0:
            raise ValueError(
                "TrainAggregator should only be used on the start of a timeseries"
            )
        self._loss += loss
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {f"{label}/mean/loss": self._loss / self._n_batches}
        dist = Distributed.get_instance()
        for key in sorted(logs.keys()):
            logs[key] = float(dist.reduce_mean(logs[key].detach()).cpu().numpy())
        return logs
