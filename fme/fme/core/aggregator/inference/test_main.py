from fme.core.aggregator.inference import InferenceAggregator
from fme.core.device import get_device

import torch


def test_labels_exist():
    agg = InferenceAggregator(record_step_20=True)
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    loss = 1.0
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    target_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    agg.record_batch(loss, target_data, gen_data, target_data_norm, gen_data_norm)
    logs = agg.get_logs(label="test")
    assert "test/mean/series" in logs
    assert "test/mean_norm/series" in logs
    assert "test/mean_step_20/l1/a" in logs
    assert "test/mean_step_20/area_weighted_rmse/a" in logs
    assert "test/mean_step_20/area_weighted_bias/a" in logs
    assert (
        "test/mean_step_20/area_weighted_mean_gradient_magnitude_percent_diff/a" in logs
    )
    table = logs["test/mean/series"]
    assert table.columns == [
        "rollout_step",
        "area_weighted_bias/a",
        "area_weighted_mean_gen/a",
        "area_weighted_mean_gradient_magnitude_percent_diff/a",
        "area_weighted_rmse/a",
    ]
    assert "test/time_mean/rmse/a" in logs
    assert "test/time_mean/bias/a" in logs
