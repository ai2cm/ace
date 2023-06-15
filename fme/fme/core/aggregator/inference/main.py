from typing import Mapping, Protocol, Dict

from .time_mean import TimeMeanAggregator
from .reduced import MeanAggregator
from ..one_step.reduced import MeanAggregator as OneStepMeanAggregator

import torch


class _Aggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
    ):
        ...

    @torch.no_grad()
    def get_logs(self, label: str):
        ...


class InferenceAggregator:
    """
    Aggregates statistics for inference.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(self, record_step_20: bool = False):
        self._aggregators: Dict[str, _Aggregator] = {
            "mean": MeanAggregator(target="denorm"),
            "mean_norm": MeanAggregator(target="norm"),
            "time_mean": TimeMeanAggregator(),
        }
        if record_step_20:
            self._aggregators["mean_step_20"] = OneStepMeanAggregator(target_time=20)

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
        for aggregator in self._aggregators.values():
            aggregator.record_batch(
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
        for name, aggregator in self._aggregators.items():
            logs.update(aggregator.get_logs(label=name))
        logs = {f"{label}/{key}": val for key, val in logs.items()}
        return logs
