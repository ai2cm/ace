import numpy as np
import torch

import fme
from fme.ace.aggregator.inference.reduced import (
    AreaWeightedReducedMetric,
    SingleTargetMeanAggregator,
)
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.typing_ import TensorDict, TensorMapping


def test_area_weighted_reduced_metric_includes_later_window_starts():
    """
    The area weighted reduced metric should assume that the start
    of a window is always recorded, as we clip it before calling.
    """

    def compute_metric(
        truth: TensorMapping, predicted: TensorMapping, weights=None, dim=()
    ) -> TensorDict:
        return {name: truth[name].mean(dim=(2, 3)) for name in truth}

    metric = AreaWeightedReducedMetric(
        device=get_device(),
        compute_metric=compute_metric,
        n_timesteps=7,
    )

    data = torch.ones([2, 3, 4, 4], device=get_device())
    data_dict = {"var1": data}
    metric.record(data_dict, data_dict, 0)
    data_nan = data.clone()
    data_nan[:, 0, :, :] = np.nan
    data_dict_nan = {"var1": data_nan}
    metric.record(data_dict_nan, data_dict_nan, 2)
    metric.record(data_dict_nan, data_dict_nan, 4)
    result_dict = metric.get()
    assert "var1" in result_dict
    result = result_dict["var1"].cpu().numpy()
    # assert tensor is all ones
    assert np.sum(np.isnan(result)) == 2
    assert np.isnan(result[2])
    assert np.isnan(result[4])


def test_single_target_mean_aggregator():
    """
    The area weighted reduced metric should assume that the start
    of a window is always recorded, as we clip it before calling.
    """
    n_sample = 10
    n_time_per_window = 22
    n_window = 3
    nx = 2
    ny = 2
    area_weights = torch.ones(ny).to(fme.get_device())
    torch.manual_seed(0)

    agg = SingleTargetMeanAggregator(
        gridded_operations=LatLonOperations(area_weights),
        n_timesteps=n_time_per_window * n_window,
    )
    data_a = torch.randn(n_sample, n_time_per_window, nx, ny, device=get_device())
    for i in range(n_window):
        data = {"a": data_a[:, i * n_time_per_window : (i + 1) * n_time_per_window]}
        agg.record_batch(data=data, i_time_start=i * n_time_per_window)

    logs = agg.get_logs(label="test")
    assert "test/series" in logs
    ds = agg.get_dataset()
    for i in range(1, data_a.shape[1]):
        raw_variable = data_a[:, i]
        raw_global_mean = raw_variable.mean().cpu().numpy()
        raw_global_std = (
            raw_variable.std(dim=(1, 2), correction=0).mean().cpu().numpy()
        )  # metrics are mean over batch
        np.testing.assert_allclose(
            raw_global_std,
            ds["weighted_std_gen-a"].isel(forecast_step=i).values.item(),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            raw_global_mean,
            ds["weighted_mean_gen-a"].isel(forecast_step=i).values.item(),
            rtol=1e-5,
        )
