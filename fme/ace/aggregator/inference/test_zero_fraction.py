import pytest
import torch

from fme.core.gridded_ops import LatLonOperations

from .data import InferenceBatchData
from .zero_fraction import ZeroFractionAggregator, ZeroFractionMetricConfig

SHAPE = (2, 3, 4, 8)  # (sample, time, lat, lon)


def _ops() -> LatLonOperations:
    return LatLonOperations(area_weights=torch.ones(SHAPE[-2:]))


def test_zero_fraction_counts_exactly_zero_cells():
    # half the lon points are exactly zero -> fraction 0.5 with uniform area
    pred = torch.ones(SHAPE)
    pred[..., : SHAPE[-1] // 2] = 0.0
    target = torch.ones(SHAPE)  # no zero cells in target
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        target={"PRATEsfc": target},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = ZeroFractionAggregator(area_weighted_mean=_ops().area_weighted_mean)
    agg.record_batch(data)
    logs = agg.get_logs("val")
    assert logs["val/gen/PRATEsfc"] == 0.5
    assert logs["val/gen_minus_target/PRATEsfc"] == 0.5
    # the standalone target fraction is not reported
    assert "val/target/PRATEsfc" not in logs


def test_zero_fraction_no_target():
    pred = torch.zeros(SHAPE)  # everything at/below zero
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = ZeroFractionAggregator(area_weighted_mean=_ops().area_weighted_mean)
    agg.record_batch(data)
    logs = agg.get_logs("")
    assert logs["gen/PRATEsfc"] == 1.0
    assert "target/PRATEsfc" not in logs
    assert "gen_minus_target/PRATEsfc" not in logs


def test_zero_fraction_threshold():
    pred = torch.full(SHAPE, 0.5)
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    # threshold 0.0: nothing at/below; threshold 1.0: everything at/below
    agg_zero = ZeroFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean, threshold=0.0
    )
    agg_zero.record_batch(data)
    assert agg_zero.get_logs("")["gen/PRATEsfc"] == 0.0
    agg_one = ZeroFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean, threshold=1.0
    )
    agg_one.record_batch(data)
    assert agg_one.get_logs("")["gen/PRATEsfc"] == 1.0


def test_zero_fraction_per_variable_threshold():
    # "a" uses the per-variable threshold (1.0 -> all below), "b" falls back to
    # the global threshold (0.0 -> none below).
    pred = {"a": torch.full(SHAPE, 0.5), "b": torch.full(SHAPE, 0.5)}
    data = InferenceBatchData(
        prediction=pred,
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = ZeroFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean,
        threshold=0.0,
        per_variable_threshold={"a": 1.0},
    )
    agg.record_batch(data)
    logs = agg.get_logs("")
    assert logs["gen/a"] == 1.0
    assert logs["gen/b"] == 0.0


def test_zero_fraction_maps_disabled_by_default():
    pred = torch.ones(SHAPE)
    pred[..., : SHAPE[-1] // 2] = 0.0
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        target={"PRATEsfc": torch.ones(SHAPE)},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = ZeroFractionAggregator(area_weighted_mean=_ops().area_weighted_mean)
    agg.record_batch(data)
    logs = agg.get_logs("")
    assert not any("map" in key for key in logs)
    assert len(agg.get_dataset().data_vars) == 0


def test_zero_fraction_maps_logged_with_target():
    # first half of lon is exactly zero in gen; target has no zero cells.
    pred = torch.ones(SHAPE)
    pred[..., : SHAPE[-1] // 2] = 0.0
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        target={"PRATEsfc": torch.ones(SHAPE)},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = ZeroFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean, include_maps=True
    )
    agg.record_batch(data)
    logs = agg.get_logs("val")
    # scalar metrics still present
    assert logs["val/gen/PRATEsfc"] == 0.5
    # side-by-side comparison and error maps logged (as wandb Images)
    assert "val/gen_target_map/PRATEsfc" in logs
    assert "val/error_map/PRATEsfc" in logs
    assert "val/gen_map/PRATEsfc" not in logs  # only when target absent

    ds = agg.get_dataset()
    gen_map = ds["gen_map-PRATEsfc"].values
    target_map = ds["target_map-PRATEsfc"].values
    error_map = ds["error_map-PRATEsfc"].values
    half = SHAPE[-1] // 2
    # per-cell fraction of time at/below 0: 1 where gen is zero, else 0
    assert (gen_map[:, :half] == 1.0).all()
    assert (gen_map[:, half:] == 0.0).all()
    assert (target_map == 0.0).all()
    assert (error_map == gen_map).all()


def test_zero_fraction_maps_no_target():
    pred = torch.zeros(SHAPE)
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = ZeroFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean, include_maps=True
    )
    agg.record_batch(data)
    logs = agg.get_logs("")
    assert "gen_map/PRATEsfc" in logs
    assert "gen_target_map/PRATEsfc" not in logs
    assert "error_map/PRATEsfc" not in logs
    ds = agg.get_dataset()
    assert (ds["gen_map-PRATEsfc"].values == 1.0).all()
    assert "target_map-PRATEsfc" not in ds
    assert "error_map-PRATEsfc" not in ds


def test_zero_fraction_metric_config_include_maps_default_false():
    assert ZeroFractionMetricConfig().include_maps is False


def test_zero_fraction_metric_config_disabled_by_default():
    config = ZeroFractionMetricConfig()
    assert config.enabled is False
    assert config.get_name() == "zero_threshold_fraction"


def test_zero_fraction_metric_config_enabled_requires_variables():
    with pytest.raises(ValueError, match="no variables"):
        ZeroFractionMetricConfig(enabled=True)
    # enabled with variables is fine
    ZeroFractionMetricConfig(enabled=True, variables=["PRATEsfc"])
