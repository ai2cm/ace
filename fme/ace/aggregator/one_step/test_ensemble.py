import pytest
import torch

from fme.ace.aggregator.one_step.ensemble import (
    CRPSMetric,
    EnsembleMeanRMSEMetric,
    SSRBiasMetric,
)


def get_tensor(shape):
    return torch.randn(*shape)


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
