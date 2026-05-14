import torch

from fme.ace.aggregator.tendency_variance import TendencyVarianceAccumulator
from fme.core.device import get_device


def test_ratio_is_one_for_identical_data():
    acc = TendencyVarianceAccumulator()
    n_sample, n_time, h, w = 4, 3, 5, 5
    data = {"a": torch.randn(n_sample, n_time, h, w, device=get_device())}
    acc.record(data, data)
    ratios = acc.get_ratios()
    assert abs(ratios["a"] - 1.0) < 1e-5


def test_ratio_reflects_variance_scaling():
    """If gen tendencies have 2x the amplitude, ratio should be ~4."""
    torch.manual_seed(42)
    acc = TendencyVarianceAccumulator()
    n_sample, n_time, h, w = 8, 4, 10, 10
    device = get_device()

    target = {"a": torch.randn(n_sample, n_time, h, w, device=device)}
    gen = {"a": 2.0 * target["a"]}

    acc.record(gen, target)
    ratios = acc.get_ratios()
    assert abs(ratios["a"] - 4.0) < 0.1


def test_multiple_variables():
    torch.manual_seed(0)
    acc = TendencyVarianceAccumulator()
    n_sample, n_time, h, w = 4, 3, 6, 6
    device = get_device()

    target = {
        "u": torch.randn(n_sample, n_time, h, w, device=device),
        "v": torch.randn(n_sample, n_time, h, w, device=device),
    }
    gen = {
        "u": 0.5 * target["u"],
        "v": 3.0 * target["v"],
    }

    acc.record(gen, target)
    ratios = acc.get_ratios()
    assert set(ratios.keys()) == {"u", "v"}
    assert abs(ratios["u"] - 0.25) < 0.05
    assert abs(ratios["v"] - 9.0) < 0.5


def test_accumulation_across_batches():
    """Ratio is stable across two identical batches."""
    torch.manual_seed(7)
    acc = TendencyVarianceAccumulator()
    n_sample, n_time, h, w = 4, 3, 8, 8
    device = get_device()

    target = {"a": torch.randn(n_sample, n_time, h, w, device=device)}
    gen = {"a": 1.5 * target["a"]}

    acc.record(gen, target)
    ratios_one = acc.get_ratios()

    acc.record(gen, target)
    ratios_two = acc.get_ratios()

    assert abs(ratios_one["a"] - ratios_two["a"]) < 1e-5


def test_skips_single_timestep():
    """Variables with only one timestep are silently skipped."""
    acc = TendencyVarianceAccumulator()
    n_sample, h, w = 4, 5, 5
    data = {"a": torch.randn(n_sample, 1, h, w, device=get_device())}
    acc.record(data, data)
    assert acc.get_ratios() == {}


def test_skips_missing_target_variable():
    acc = TendencyVarianceAccumulator()
    n_sample, n_time, h, w = 2, 3, 4, 4
    device = get_device()
    gen = {"a": torch.randn(n_sample, n_time, h, w, device=device)}
    target = {"b": torch.randn(n_sample, n_time, h, w, device=device)}
    acc.record(gen, target)
    assert acc.get_ratios() == {}


def test_get_logs_keys():
    acc = TendencyVarianceAccumulator()
    n_sample, n_time, h, w = 2, 3, 4, 4
    data = {"x": torch.randn(n_sample, n_time, h, w, device=get_device())}
    acc.record(data, data)
    logs = acc.get_logs("my_label")
    assert "my_label/tendency_variance_ratio/x" in logs
