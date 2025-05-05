import numpy as np
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping

from .reduced_metrics import AreaWeightedReducedMetric, ReducedMetric


def get_gen_shape(gen_data: TensorMapping):
    for name in gen_data:
        return gen_data[name].shape


class MeanAggregator:
    """
    Aggregator for mean-reduced metrics.

    These are metrics such as means which reduce to a single float for each batch,
    and then can be averaged across batches to get a single float for the
    entire dataset. This is important because the aggregator uses the mean to combine
    metrics across batches and processors.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        target_time: int = 1,
    ):
        self._gridded_operations = gridded_operations
        self._n_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())
        self._target_time = target_time
        self._dist = Distributed.get_instance()

        device = get_device()
        self._variable_metrics: dict[str, ReducedMetric] = {}
        self._variable_metrics["weighted_rmse"] = AreaWeightedReducedMetric(
            device=device,
            compute_metric=self._gridded_operations.area_weighted_rmse_dict,
        )
        self._variable_metrics["weighted_bias"] = AreaWeightedReducedMetric(
            device=device,
            compute_metric=self._gridded_operations.area_weighted_mean_bias_dict,
        )
        self._variable_metrics["weighted_grad_mag_percent_diff"] = (
            AreaWeightedReducedMetric(
                device=device,
                compute_metric=self._gridded_operations.area_weighted_gradient_magnitude_percent_diff_dict,  # noqa: E501
            )
        )

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        loss: torch.Tensor = torch.tensor(np.nan),
        i_time_start: int = 0,
    ):
        self._loss += loss
        time_dim = 1
        time_len = gen_data[list(gen_data.keys())[0]].shape[time_dim]
        target_time = self._target_time - i_time_start
        if target_time >= 0 and time_len > target_time:
            target_snapshot = {}
            gen_snapshot = {}
            for name in gen_data.keys():
                target_snapshot[name] = target_data[name].select(
                    dim=time_dim, index=target_time
                )
                gen_snapshot[name] = gen_data[name].select(
                    dim=time_dim, index=target_time
                )
            for metric in self._variable_metrics.values():
                metric.record(
                    target=target_snapshot,
                    gen=gen_snapshot,
                )
            # only increment n_batches if we actually recorded a batch
            self._n_batches += 1

    def _get_data(self):
        if self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        data: dict[str, torch.Tensor] = {"loss": self._loss / self._n_batches}
        for name, metric in self._variable_metrics.items():
            metric_results = metric.get()  # TensorDict: {var_name: metric_data}
            for key in metric_results:
                data[f"{name}/{key}"] = metric_results[key] / self._n_batches
        for key in sorted(data.keys()):
            data[key] = float(self._dist.reduce_mean(data[key].detach()).cpu().numpy())
        return data

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return {
            f"{label}/{key}": data for key, data in sorted(self._get_data().items())
        }

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        data = self._get_data()
        data = {key.replace("/", "-"): data[key] for key in data}
        data_vars = {}
        for key, value in data.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)
