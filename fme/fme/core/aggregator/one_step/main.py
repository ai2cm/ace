from typing import Mapping

import torch

from .reduced import MeanAggregator
from .snapshot import SnapshotAggregator


class OneStepAggregator:
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(self, area_weights: torch.Tensor):
        self._snapshot = SnapshotAggregator()
        self._mean = MeanAggregator(area_weights)

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
    ):
        if len(target_data) == 0:
            raise ValueError("No data in target_data")
        if len(gen_data) == 0:
            raise ValueError("No data in gen_data")
        for aggregator in (
            self._snapshot,
            self._mean,
        ):
            aggregator.record_batch(  # type: ignore
                loss=loss,
                target_data=target_data,
                gen_data=gen_data,
                target_data_norm=target_data_norm,
                gen_data_norm=gen_data_norm,
            )

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = self._mean.get_logs(label="mean")
        logs.update(self._snapshot.get_logs(label="snapshot"))
        logs = {f"{label}/{key}": val for key, val in logs.items()}
        return logs
