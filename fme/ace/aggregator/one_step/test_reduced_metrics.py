import pytest
import torch

from fme import get_device
from fme.ace.aggregator.one_step.reduced_metrics import AreaWeightedReducedMetric
from fme.core.typing_ import TensorDict, TensorMapping

DEVICE = get_device()


def mock_area_weighted_metric(
    truth: TensorMapping, predicted: TensorMapping
) -> TensorDict:
    result = {}
    for name in truth:
        diff = predicted[name] - truth[name]
        result[name] = diff.mean(dim=list(range(1, diff.ndim)))
    return result


def test_area_weighted_reduced_metric_record_and_get():
    metric = AreaWeightedReducedMetric(
        device=DEVICE, compute_metric=mock_area_weighted_metric
    )

    # Batch 1
    target1 = {
        "a": torch.ones(2, 3, 4, 5, device=DEVICE),
        "b": torch.zeros(2, 3, 4, 5, device=DEVICE),
    }
    gen1 = {
        "a": torch.full((2, 3, 4, 5), 2.0, device=DEVICE),
        "b": torch.ones(2, 3, 4, 5, device=DEVICE),
    }
    # Expected batch 1 means: a=(2-1)=1, b=(1-0)=1
    metric.record(target1, gen1)

    # Batch 2
    target2 = {
        "a": torch.ones(2, 3, 4, 5, device=DEVICE),
        "b": torch.zeros(2, 3, 4, 5, device=DEVICE),
    }
    gen2 = {
        "a": torch.full((2, 3, 4, 5), 4.0, device=DEVICE),
        "b": torch.full((2, 3, 4, 5), 2.0, device=DEVICE),
    }
    # Expected batch 2 means: a=(4-1)=3, b=(2-0)=2
    metric.record(target2, gen2)

    # AreaWeightedReducedMetric sums the batch averages
    # Expected total: a = 1 + 3, b = 1 + 2
    expected_total = {
        "a": torch.tensor(1.0 + 3.0, device=DEVICE),
        "b": torch.tensor(1.0 + 2.0, device=DEVICE),
    }

    result = metric.get()

    assert result.keys() == expected_total.keys()
    torch.testing.assert_close(result["a"], expected_total["a"])
    torch.testing.assert_close(result["b"], expected_total["b"])


def test_area_weighted_reduced_metric_new_keys_raises():
    metric = AreaWeightedReducedMetric(
        device=DEVICE, compute_metric=mock_area_weighted_metric
    )

    target1 = {"a": torch.ones(1, 1, device=DEVICE)}
    gen1 = {"a": torch.ones(1, 1, device=DEVICE)}
    metric.record(target1, gen1)

    target2 = {
        "a": torch.ones(1, 1, device=DEVICE),
        "b": torch.ones(1, 1, device=DEVICE),
    }
    gen2 = {"a": torch.ones(1, 1, device=DEVICE), "b": torch.ones(1, 1, device=DEVICE)}
    with pytest.raises(ValueError, match="'b'"):
        metric.record(target2, gen2)


def test_area_weighted_reduced_metric_missing_keys_raises():
    metric = AreaWeightedReducedMetric(
        device=DEVICE, compute_metric=mock_area_weighted_metric
    )

    target1 = {
        "a": torch.ones(1, 1, device=DEVICE),
        "b": torch.ones(1, 1, device=DEVICE),
    }
    gen1 = {"a": torch.ones(1, 1, device=DEVICE), "b": torch.ones(1, 1, device=DEVICE)}
    metric.record(target1, gen1)

    target2 = {"a": torch.ones(1, 1, device=DEVICE)}
    gen2 = {"a": torch.ones(1, 1, device=DEVICE)}
    with pytest.raises(ValueError, match="'b'"):
        metric.record(target2, gen2)


def test_area_weighted_reduced_metric_empty():
    metric = AreaWeightedReducedMetric(
        device=DEVICE, compute_metric=mock_area_weighted_metric
    )
    metrics = metric.get()
    assert torch.isnan(metrics["anything"])
