import pytest

from fme.core.generics.metrics_aggregator import MetricsAggregator


@pytest.mark.parametrize(
    "metrics, expected",
    [
        (
            [
                {},
            ],
            {},
        ),
        (
            [
                {"loss": 1.0},
            ],
            {"loss": 1.0},
        ),
        (
            [{"loss": 1.0}, {"loss": 2.0}, {"loss": 3.0}],
            {"loss": 2.0},
        ),
        (
            [{"loss": 1.0}, {"another_metric": 5.0}, {"loss": 3.0}],
            {"loss": 2.0, "another_metric": 5.0},
        ),
    ],
)
def test_metrics_aggregator(
    metrics: list[dict[str, float]], expected: dict[str, float]
):
    aggregator = MetricsAggregator()
    for metric in metrics:
        aggregator.record(metric)
    assert aggregator.get_metrics() == expected


def test_metrics_aggregator_clear():
    aggregator = MetricsAggregator()
    aggregator.record({"loss": 1.0})
    aggregator.clear()
    assert aggregator.get_metrics() == {}
    aggregator.record({"another_metric": 5.0})
    assert aggregator.get_metrics() == {"another_metric": 5.0}
    aggregator.clear()
    assert aggregator.get_metrics() == {}
    aggregator.record({"loss": 2.0})
    assert aggregator.get_metrics() == {"loss": 2.0}
