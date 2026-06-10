import pytest
import torch

from fme.ace.aggregator.one_step.ensemble import (
    CRPSMetric,
    EnsembleMeanRMSEMetric,
    SSRBiasMetric,
    _EnsembleAggregator,
)
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.typing_ import EnsembleTensorDict


def get_tensor(shape):
    return torch.randn(*shape)


def _make_ensemble_batch(names, shape=(2, 3, 1, 4, 4)):
    """Make an EnsembleTensorDict of given shape for each named variable."""
    return EnsembleTensorDict(
        {name: torch.randn(*shape, device=get_device()) for name in names}
    )


def test_crps_metric_gives_correct_shape():
    metric = CRPSMetric()
    n_batch = 10
    n_sample = 2
    n_time = 3
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)


def test_ssr_metric_gives_correct_shape():
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 10
    n_sample = 2
    n_time = 3
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)


@pytest.mark.parametrize("n_sample", [2, 5, 10])
def test_ssr_bias_metric_unbiased(n_sample):
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 5000
    n_time = 3
    n_y = 4
    n_x = 5
    # identical distribution for target and gen
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)
    torch.testing.assert_close(got.mean(), torch.tensor(0.0), atol=1e-2, rtol=0.0)


@pytest.mark.parametrize("n_sample", [2, 5, 10])
def test_ssr_bias_metric_doubled_spread(n_sample):
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 5000
    n_time = 3
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x)) * 2
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)
    torch.testing.assert_close(got.mean(), torch.tensor(1.0), atol=1e-2, rtol=0.0)


def test_ssr_finite_with_small_ensemble():
    """SSR must be finite for all grid cells, even with n_ensemble=2 and
    few batches, which previously produced NaN via sqrt of negative
    unbiased_mse."""
    torch.manual_seed(42)
    metric = SSRBiasMetric()
    n_batch = 3
    n_sample = 2
    n_time = 1
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = target + 0.01 * get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert torch.isfinite(got).all(), f"Non-finite SSR values: {got}"


def test_ssr_nontrivial_with_differing_members():
    """When ensemble members actually differ, SSR should not be -1."""
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 500
    n_sample = 5
    n_time = 1
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert torch.isfinite(got).all()
    assert (got > -0.9).all(), f"SSR unexpectedly near -1: mean={got.mean():.3f}"


def test_ssr_identical_members_gives_negative_one():
    """When all ensemble members are identical, SSR should be -1
    (zero spread / nonzero skill)."""
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 10
    n_sample = 3
    n_time = 1
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    single_pred = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = single_pred.expand(n_batch, n_sample, n_time, n_y, n_x).clone()
    metric.record(target=target, gen=gen)
    got = metric.get()
    torch.testing.assert_close(got, torch.full_like(got, -1.0), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("n_sample", [2, 10])
def test_ensemble_mean_metric(n_sample):
    # this simple check only works for even n_sample
    torch.manual_seed(0)
    metric = EnsembleMeanRMSEMetric()
    n_batch = 5000
    n_time = 3
    n_y = 4
    n_x = 5
    target = torch.ones((n_batch, 1, n_time, n_y, n_x))
    gen = torch.ones((n_batch, n_sample, n_time, n_y, n_x)) * 2
    # half the samples are 0
    gen[:, : n_sample // 2, ...] = 0
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)
    torch.testing.assert_close(got.mean(), torch.tensor(0.0), atol=1e-2, rtol=0.0)


def test_aggregator_denorm_does_not_log_channel_mean():
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="denorm",
    )
    names = ["a", "b"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)
    agg.record_batch(target_data=target, gen_data=gen)
    logs = agg.get_logs(label="metrics")
    for metric in ("crps", "ssr_bias", "ensemble_mean_rmse"):
        assert f"metrics/{metric}/a" in logs
        assert f"metrics/{metric}/b" in logs
        assert f"metrics/{metric}/channel_mean" not in logs


def test_aggregator_norm_requires_norm_data():
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
    )
    target = _make_ensemble_batch(["a"])
    gen = _make_ensemble_batch(["a"])
    with pytest.raises(ValueError, match="target_data_norm and gen_data_norm"):
        agg.record_batch(target_data=target, gen_data=gen)


def test_aggregator_norm_logs_channel_mean_all_variables():
    """When channel_mean_names is None and target='norm', channel mean is
    computed over all variables."""
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
    )
    names = ["a", "b", "c"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)
    agg.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    logs = agg.get_logs(label="metrics")
    for metric in ("crps", "ssr_bias", "ensemble_mean_rmse"):
        assert f"metrics/{metric}/channel_mean" in logs
        expected = sum(logs[f"metrics/{metric}/{n}"] for n in names) / len(names)
        torch.testing.assert_close(
            torch.tensor(float(logs[f"metrics/{metric}/channel_mean"])),
            torch.tensor(float(expected)),
            atol=1e-6,
            rtol=1e-6,
        )


def test_aggregator_norm_logs_channel_mean_subset():
    """channel_mean_names restricts the channel mean to the listed variables."""
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
        channel_mean_names=["a", "c"],
    )
    names = ["a", "b", "c"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)
    agg.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    logs = agg.get_logs(label="metrics")
    for metric in ("crps", "ssr_bias", "ensemble_mean_rmse"):
        expected = (logs[f"metrics/{metric}/a"] + logs[f"metrics/{metric}/c"]) / 2.0
        torch.testing.assert_close(
            torch.tensor(float(logs[f"metrics/{metric}/channel_mean"])),
            torch.tensor(float(expected)),
            atol=1e-6,
            rtol=1e-6,
        )


def test_aggregator_report_variables_filters_per_variable_but_keeps_channel_mean():
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    names = ["a", "b", "c"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)

    agg_full = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
        channel_mean_names=names,
    )
    agg_full.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    full_logs = agg_full.get_logs(label="metrics")

    agg_filtered = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
        channel_mean_names=names,
        report_variables=["a"],
    )
    agg_filtered.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    filtered_logs = agg_filtered.get_logs(label="metrics")

    for metric in ("crps", "ssr_bias", "ensemble_mean_rmse"):
        assert f"metrics/{metric}/a" in filtered_logs
        assert f"metrics/{metric}/b" not in filtered_logs
        assert f"metrics/{metric}/c" not in filtered_logs
        assert f"metrics/{metric}/channel_mean" in filtered_logs
        torch.testing.assert_close(
            torch.tensor(float(filtered_logs[f"metrics/{metric}/channel_mean"])),
            torch.tensor(float(full_logs[f"metrics/{metric}/channel_mean"])),
            atol=1e-6,
            rtol=1e-6,
        )


def test_aggregator_norm_raises_on_unknown_channel_mean_names():
    """Names in channel_mean_names that aren't in the data raise KeyError."""
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
        channel_mean_names=["a", "not_present"],
    )
    names = ["a", "b"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)
    agg.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    with pytest.raises(KeyError, match="not_present"):
        agg.get_logs(label="metrics")
