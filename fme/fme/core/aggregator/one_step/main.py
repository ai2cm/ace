from typing import Mapping, Optional, Protocol

import torch

from fme.core.aggregator.one_step.derived import DerivedMetricsAggregator
from fme.core.data_loading.data_typing import SigmaCoordinates, VariableMetadata

from .map import MapAggregator
from .reduced import MeanAggregator
from .snapshot import SnapshotAggregator


class _Aggregator(Protocol):
    def get_logs(self, label: str) -> Mapping[str, torch.Tensor]:
        ...

    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
    ) -> None:
        ...


class OneStepAggregator:
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ):
        self._aggregators: Mapping[str, _Aggregator] = {
            "snapshot": SnapshotAggregator(metadata),
            "mean": MeanAggregator(area_weights),
            "derived": DerivedMetricsAggregator(area_weights, sigma_coordinates),
            "mean_map": MapAggregator(metadata),
        }

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

        for agg in self._aggregators.values():
            agg.record_batch(
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
        logs = {}
        for agg_label in self._aggregators:
            for k, v in self._aggregators[agg_label].get_logs(label=agg_label).items():
                logs[f"{label}/{k}"] = v
        return logs
