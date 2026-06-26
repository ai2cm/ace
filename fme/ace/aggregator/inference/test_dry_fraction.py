import torch

from fme.core.gridded_ops import LatLonOperations

from .data import InferenceBatchData
from .dry_fraction import DryFractionAggregator, DryFractionMetricConfig

SHAPE = (2, 3, 4, 8)  # (sample, time, lat, lon)


def _ops() -> LatLonOperations:
    return LatLonOperations(area_weights=torch.ones(SHAPE[-2:]))


def test_dry_fraction_counts_exactly_zero_cells():
    # half the lon points are exactly zero -> dry fraction 0.5 with uniform area
    pred = torch.ones(SHAPE)
    pred[..., : SHAPE[-1] // 2] = 0.0
    target = torch.ones(SHAPE)  # no dry cells in target
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        target={"PRATEsfc": target},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = DryFractionAggregator(area_weighted_mean=_ops().area_weighted_mean)
    agg.record_batch(data)
    logs = agg.get_logs("val")
    assert logs["val/dry_fraction/gen/PRATEsfc"] == 0.5
    assert logs["val/dry_fraction/target/PRATEsfc"] == 0.0
    assert logs["val/dry_fraction/gen_minus_target/PRATEsfc"] == 0.5


def test_dry_fraction_no_target():
    pred = torch.zeros(SHAPE)  # everything dry
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = DryFractionAggregator(area_weighted_mean=_ops().area_weighted_mean)
    agg.record_batch(data)
    logs = agg.get_logs("")
    assert logs["dry_fraction/gen/PRATEsfc"] == 1.0
    assert "dry_fraction/target/PRATEsfc" not in logs
    assert "dry_fraction/gen_minus_target/PRATEsfc" not in logs


def test_dry_fraction_threshold():
    pred = torch.full(SHAPE, 0.5)
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    # threshold 0.0: nothing dry; threshold 1.0: everything dry
    agg_zero = DryFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean, threshold=0.0
    )
    agg_zero.record_batch(data)
    assert agg_zero.get_logs("")["dry_fraction/gen/PRATEsfc"] == 0.0
    agg_one = DryFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean, threshold=1.0
    )
    agg_one.record_batch(data)
    assert agg_one.get_logs("")["dry_fraction/gen/PRATEsfc"] == 1.0


def test_dry_fraction_metric_config_disabled_by_default():
    config = DryFractionMetricConfig()
    assert config.enabled is False
    assert config.get_name() == "dry_fraction"
