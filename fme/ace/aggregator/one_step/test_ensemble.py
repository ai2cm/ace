import math

import pytest
import torch

from fme.ace.aggregator.one_step.ensemble import (
    CRPSMetric,
    EnsembleMeanRMSEMetric,
    SSRBiasL1Metric,
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


def test_ssr_prescribed_cell_is_zero_but_zero_spread_with_error_is_negative_one():
    """A prescribed cell (members identical and equal to the target) is a 0/0
    and reports 0, while a zero-spread cell with nonzero error still gives -1."""
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch, n_sample, n_time, n_y, n_x = 10, 3, 1, 2, 2
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    # zero spread everywhere: every ensemble member identical
    single_pred = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = single_pred.expand(n_batch, n_sample, n_time, n_y, n_x).clone()
    # left column: members also equal the target -> zero skill too (prescribed)
    gen[..., 0] = target[..., 0]
    metric.record(target=target, gen=gen)
    got = metric.get()
    torch.testing.assert_close(
        got[..., 0], torch.zeros_like(got[..., 0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        got[..., 1], torch.full_like(got[..., 1], -1.0), atol=1e-6, rtol=0.0
    )


def test_aggregator_ssr_bias_prescribed_cells_do_not_pull_scalar_to_negative_one():
    """Prescribed cells contribute 0, not the -1 floor, so a mostly-prescribed
    field is not dragged toward -1. With uniform weights the scalar equals the
    plain mean of the per-cell field."""
    torch.manual_seed(0)
    n_batch, n_sample, n_time, n_y, n_x = 50, 4, 1, 2, 4
    area_weights = torch.ones([n_y, n_x], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="denorm",
    )
    target = torch.randn(n_batch, 1, n_time, n_y, n_x, device=get_device())
    gen = torch.randn(n_batch, n_sample, n_time, n_y, n_x, device=get_device())
    # left half of the grid prescribed: every member equals the target
    gen[..., :2] = target[..., :2]
    agg.record_batch(
        target_data=EnsembleTensorDict({"a": target}),
        gen_data=EnsembleTensorDict({"a": gen}),
    )
    scalar = float(agg.get_logs(label="metrics")["metrics/ssr_bias/a"])
    assert math.isfinite(scalar)
    # not pinned to the -1 floor by the prescribed half
    assert scalar > -0.5, scalar

    # with uniform weights the scalar is the plain mean of the per-cell field,
    # where the prescribed cells contribute 0 (not -1)
    field = SSRBiasMetric()
    field.record(target=target, gen=gen)
    expected = float(field.get().mean())
    assert math.isclose(scalar, expected, rel_tol=1e-5, abs_tol=1e-5), (
        scalar,
        expected,
    )


def test_ssr_bias_l1_metric_gives_correct_shape():
    torch.manual_seed(0)
    metric = SSRBiasL1Metric()
    n_batch, n_sample, n_time, n_y, n_x = 10, 3, 3, 4, 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)


@pytest.mark.parametrize("n_sample", [2, 5, 10])
def test_ssr_bias_l1_metric_calibrated_is_zero(n_sample):
    """A calibrated ensemble (members and target from the same distribution)
    has E|X - X'| = E|X - y|, so ssr_bias_l1 -> 0."""
    torch.manual_seed(0)
    metric = SSRBiasL1Metric()
    n_batch, n_time, n_y, n_x = 5000, 3, 4, 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert got.shape == (n_y, n_x)
    torch.testing.assert_close(got.mean(), torch.tensor(0.0), atol=1e-2, rtol=0.0)


@pytest.mark.parametrize("n_sample", [2, 5, 10])
def test_ssr_bias_l1_metric_underdispersed_is_negative(n_sample):
    """Members clustered far more tightly than they are from the target:
    spread_L1 << skill_L1, so ssr_bias_l1 < 0."""
    torch.manual_seed(0)
    metric = SSRBiasL1Metric()
    n_batch, n_time, n_y, n_x = 5000, 3, 4, 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x)) * 0.1
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert torch.isfinite(got).all()
    assert (got < 0).all(), f"ssr_bias_l1 not under-dispersed: {got}"


def test_ssr_bias_l1_metric_forms_ratio_at_end_not_per_batch():
    """The spread and skill numerators/denominators accumulate across batches
    and the ratio forms once in get() -- distinct from averaging per-batch
    ratios. Two single-cell batches with known spread/skill make the two
    orderings numerically distinguishable."""
    metric = SSRBiasL1Metric()

    def _cell(members, obs):
        gen = torch.tensor(members, dtype=torch.float32).reshape(1, -1, 1, 1, 1)
        target = torch.tensor([[[[[float(obs)]]]]])
        return target, gen

    # batch 1: members {0, 2}, obs 5 -> spread=|0-2|=2, skill=(5+3)/2=4
    metric.record(*_cell([0.0, 2.0], 5.0))
    # batch 2: members {0, 10}, obs 0 -> spread=|0-10|=10, skill=(0+10)/2=5
    metric.record(*_cell([0.0, 10.0], 0.0))
    got = float(metric.get().reshape(()))
    # ratio of the summed terms: (2+10)/(4+5) - 1, NOT the mean of per-batch
    # ratios 0.5 * ((2/4 - 1) + (10/5 - 1)) = 0.25.
    expected = (2.0 + 10.0) / (4.0 + 5.0) - 1.0
    assert math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-6), got
    assert not math.isclose(got, 0.25, rel_tol=1e-6, abs_tol=1e-6)


def test_ssr_bias_l1_prescribed_cell_is_zero_but_zero_spread_with_error_negative():
    """A prescribed cell (members identical and equal to the target) is a 0/0
    and reports 0; a zero-spread cell with nonzero error still gives -1."""
    torch.manual_seed(0)
    metric = SSRBiasL1Metric()
    n_batch, n_sample, n_time, n_y, n_x = 10, 3, 1, 2, 2
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    # zero spread everywhere: every ensemble member identical
    single_pred = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = single_pred.expand(n_batch, n_sample, n_time, n_y, n_x).clone()
    # left column: members also equal the target -> zero skill too (prescribed)
    gen[..., 0] = target[..., 0]
    metric.record(target=target, gen=gen)
    got = metric.get()
    torch.testing.assert_close(
        got[..., 0], torch.zeros_like(got[..., 0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        got[..., 1], torch.full_like(got[..., 1], -1.0), atol=1e-6, rtol=0.0
    )


def test_aggregator_ssr_bias_l1_disabled_by_default():
    """With enable_ssr_bias_l1 unset, no ssr_bias_l1 keys appear in the logs or
    dataset, and the other families are unchanged."""
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=True,
        target="norm",
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
    logs = agg.get_logs(label="metrics")
    assert not any("ssr_bias_l1" in key for key in logs)
    dataset = agg.get_dataset()
    assert not any("ssr_bias_l1" in str(key) for key in dataset.data_vars)
    # the L2 family is still present
    assert "metrics/ssr_bias/a" in logs


def test_aggregator_ssr_bias_l1_enabled_emits_all_shapes():
    """When enabled, ssr_bias_l1 emits per-variable, mean_map, and (for
    target='norm') channel_mean entries, matching the other families."""
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=True,
        target="norm",
        enable_ssr_bias_l1=True,
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
    logs = agg.get_logs(label="metrics")
    for name in names:
        assert f"metrics/ssr_bias_l1/{name}" in logs
        assert f"metrics/ssr_bias_l1/mean_map/{name}" in logs
    assert "metrics/ssr_bias_l1/channel_mean" in logs


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


def test_aggregator_norm_channel_mean_excludes_all_nan_target():
    """A variable whose target is entirely NaN (e.g. filled by
    allow_missing_variables) is excluded from the channel mean."""
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
    # "c" is missing from the data: entirely-NaN target.
    target = EnsembleTensorDict(
        {**target, "c": torch.full_like(target["c"], torch.nan)}
    )
    agg.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    logs = agg.get_logs(label="metrics")
    for metric in ("crps", "ensemble_mean_rmse"):
        channel_mean = float(logs[f"metrics/{metric}/channel_mean"])
        assert not math.isnan(channel_mean)
        # equals the mean of the two valid-target channels (a, b); "c" excluded.
        expected = (logs[f"metrics/{metric}/a"] + logs[f"metrics/{metric}/b"]) / 2
        torch.testing.assert_close(
            torch.tensor(channel_mean),
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
