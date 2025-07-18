import torch

from fme.ace.aggregator.one_step.ensemble import CRPSMetric, SSRBiasMetric


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


def test_ssr_bias_metric_unbiased():
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 5000
    n_sample = 2
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


def test_ssr_bias_metric_doubled_spread():
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 5000
    n_sample = 2
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
