from typing import Mapping, Optional, Dict

import torch
from torch import nn
from fme.core.device import get_device
from fme.core import metrics
from fme.core.distributed import Distributed
from ..reduced_metrics import AreaWeightedReducedMetric, ReducedMetric


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class L1Loss:
    def __init__(self, device: torch.device):
        self._total = torch.tensor(0.0, device=device)

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        self._total += nn.functional.l1_loss(
            gen,
            target,
        )

    def get(self) -> torch.Tensor:
        return self._total


class MeanAggregator:
    """
    Aggregator for mean-reduced metrics.

    These are metrics such as means which reduce to a single float for each batch,
    and then can be averaged across batches to get a single float for the
    entire dataset. This is important because the aggregator uses the mean to combine
    metrics across batches and processors.
    """

    def __init__(self, target_time: int = 1):
        self._area_weights = None
        self._shape_x = None
        self._shape_y = None
        self._n_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())
        self._variable_metrics: Optional[Dict[str, Dict[str, ReducedMetric]]] = None
        self._target_time = target_time

    def _get_variable_metrics(self, gen_data: Mapping[str, torch.Tensor]):
        if self._variable_metrics is None:
            self._variable_metrics = {
                "l1": {},
                "area_weighted_rmse": {},
                "area_weighted_bias": {},
                "area_weighted_mean_gradient_magnitude_percent_diff": {},
            }
            device = get_device()
            gen_shape = get_gen_shape(gen_data)
            area_weights = self._get_area_weights(
                shape_x=gen_shape[-2],
                shape_y=gen_shape[-1],
            )
            for key in gen_data:
                self._variable_metrics["l1"][key] = L1Loss(device=device)
                self._variable_metrics["area_weighted_rmse"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=area_weights,
                    device=device,
                    compute_metric=metrics.root_mean_squared_error,
                )
                self._variable_metrics["area_weighted_bias"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=area_weights,
                    device=device,
                    compute_metric=metrics.weighted_mean_bias,
                )
                self._variable_metrics[
                    "area_weighted_mean_gradient_magnitude_percent_diff"
                ][key] = AreaWeightedReducedMetric(
                    area_weights=area_weights,
                    device=device,
                    compute_metric=metrics.gradient_magnitude_percent_diff,
                )
        return self._variable_metrics

    def _get_area_weights(self, shape_x, shape_y):
        if self._area_weights is None:
            self._area_weights = metrics.spherical_area_weights(
                shape_x,
                shape_y,
                device=get_device(),
            )
        elif self._shape_x != shape_x or self._shape_y != shape_y:
            self._area_weights = metrics.spherical_area_weights(
                shape_x,
                shape_y,
                device=get_device(),
            )
        self._shape_x = shape_x
        self._shape_y = shape_y
        return self._area_weights

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
    ):
        self._loss += loss
        variable_metrics = self._get_variable_metrics(gen_data)
        time_dim = 1
        for name in gen_data.keys():
            for metric in variable_metrics:
                variable_metrics[metric][name].record(
                    target=target_data[name].select(
                        dim=time_dim, index=self._target_time
                    ),
                    gen=gen_data[name].select(dim=time_dim, index=self._target_time),
                )
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        if self._variable_metrics is None or self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        logs = {f"{label}/loss": self._loss / self._n_batches}
        for metric in self._variable_metrics:
            for key in self._variable_metrics[metric]:
                logs[f"{label}/{metric}/{key}"] = (
                    self._variable_metrics[metric][key].get() / self._n_batches
                )
        dist = Distributed.get_instance()
        for key in sorted(logs.keys()):
            logs[key] = float(dist.reduce_mean(logs[key].detach()).cpu().numpy())
        return logs
