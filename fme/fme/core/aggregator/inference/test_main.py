import fme
from fme.core.aggregator.inference import InferenceAggregator
from fme.core.device import get_device

import torch


def test_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    loss = 1.0
    area_weights = torch.ones(ny).to(fme.get_device())
    agg = InferenceAggregator(area_weights, record_step_20=True, log_video=True)
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    target_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    agg.record_batch(loss, target_data, gen_data, target_data_norm, gen_data_norm)
    logs = agg.get_logs(label="test")
    assert "test/mean/series" in logs
    assert "test/mean_norm/series" in logs
    assert "test/mean_step_20/l1/a" in logs
    assert "test/mean_step_20/weighted_rmse/a" in logs
    assert "test/mean_step_20/weighted_bias/a" in logs
    assert "test/mean_step_20/weighted_grad_mag_percent_diff/a" in logs
    table = logs["test/mean/series"]
    assert table.columns == [
        "forecast_step",
        "weighted_bias/a",
        "weighted_grad_mag_percent_diff/a",
        "weighted_mean_gen/a",
        "weighted_rmse/a",
    ]
    assert "test/time_mean/rmse/a" in logs
    assert "test/time_mean/bias/a" in logs
    assert "test/video/a" in logs


def test_inference_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    loss = 1.0
    area_weights = torch.ones(ny).to(fme.get_device())
    agg = InferenceAggregator(area_weights, record_step_20=True, log_video=True)
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    target_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    agg.record_batch(loss, target_data, gen_data, target_data_norm, gen_data_norm)
    logs = agg.get_inference_logs(label="test")
    assert isinstance(logs, list)
    assert len(logs) == n_time
    assert "test/mean/weighted_bias/a" in logs[0]
    assert "test/mean/weighted_mean_gen/a" in logs[0]
    assert "test/mean/weighted_grad_mag_percent_diff/a" in logs[0]
    assert "test/mean/weighted_rmse/a" in logs[0]
    assert "test/mean_norm/weighted_bias/a" in logs[0]
    assert "test/mean_norm/weighted_mean_gen/a" in logs[0]
    assert "test/mean_norm/weighted_grad_mag_percent_diff/a" in logs[0]
    assert "test/mean_norm/weighted_rmse/a" in logs[0]
    # series/table data should be rolled out, not included as a table
    assert "test/mean/series" not in logs[0]
    assert "test/mean_norm/series" not in logs[0]
    assert "test/reduced/series" not in logs[0]
    assert "test/reduced_norm/series" not in logs[0]
