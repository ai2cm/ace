import pytest
import torch

from fme.core.gridded_ops import LatLonOperations

from .data import InferenceBatchData
from .near_zero_fraction import NearZeroFractionAggregator, NearZeroFractionMetricConfig

SHAPE = (2, 3, 4, 8)  # (sample, time, lat, lon)


def _ops() -> LatLonOperations:
    return LatLonOperations(area_weights=torch.ones(SHAPE[-2:]))


def test_near_zero_fraction_counts_exactly_zero_cells():
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
    agg = NearZeroFractionAggregator(area_weighted_mean=_ops().area_weighted_mean)
    agg.record_batch(data)
    logs = agg.get_logs("val")
    assert logs["val/gen/PRATEsfc"] == 0.5
    assert logs["val/gen_minus_target/PRATEsfc"] == 0.5
    # the standalone target fraction is not reported
    assert "val/target/PRATEsfc" not in logs


def test_near_zero_fraction_no_target():
    pred = torch.zeros(SHAPE)  # everything at/below zero
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = NearZeroFractionAggregator(area_weighted_mean=_ops().area_weighted_mean)
    agg.record_batch(data)
    logs = agg.get_logs("")
    assert logs["gen/PRATEsfc"] == 1.0
    assert "target/PRATEsfc" not in logs
    assert "gen_minus_target/PRATEsfc" not in logs


def test_near_zero_fraction_threshold():
    pred = torch.full(SHAPE, 0.5)
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    # threshold 0.0: nothing at/below; threshold 1.0: everything at/below
    agg_zero = NearZeroFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean, threshold=0.0
    )
    agg_zero.record_batch(data)
    assert agg_zero.get_logs("")["gen/PRATEsfc"] == 0.0
    agg_one = NearZeroFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean, threshold=1.0
    )
    agg_one.record_batch(data)
    assert agg_one.get_logs("")["gen/PRATEsfc"] == 1.0


def test_near_zero_fraction_per_variable_threshold():
    # "a" uses the per-variable threshold (1.0 -> all below), "b" falls back to
    # the global threshold (0.0 -> none below).
    pred = {"a": torch.full(SHAPE, 0.5), "b": torch.full(SHAPE, 0.5)}
    data = InferenceBatchData(
        prediction=pred,
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = NearZeroFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean,
        threshold=0.0,
        per_variable_threshold={"a": 1.0},
    )
    agg.record_batch(data)
    logs = agg.get_logs("")
    assert logs["gen/a"] == 1.0
    assert logs["gen/b"] == 0.0


def test_near_zero_fraction_maps_disabled_by_default():
    pred = torch.ones(SHAPE)
    pred[..., : SHAPE[-1] // 2] = 0.0
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        target={"PRATEsfc": torch.ones(SHAPE)},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = NearZeroFractionAggregator(area_weighted_mean=_ops().area_weighted_mean)
    agg.record_batch(data)
    logs = agg.get_logs("")
    assert not any("map" in key for key in logs)
    assert len(agg.get_dataset().data_vars) == 0


def test_near_zero_fraction_maps_logged_with_target():
    # Construct a gen field with a distinct lat-row pattern, lon-col pattern,
    # a partial-in-time cell, and an initial-condition-only cell, so the test
    # pins the [lat, lon] orientation, the sample/time normalization, and the
    # i_time_start==0 initial-condition drop. SHAPE has 3 timesteps; with the
    # IC dropped, 2 timesteps (t=1, t=2) remain. target has no zero cells.
    pred = torch.ones(SHAPE)  # (sample, time, lat, lon)
    pred[:, :, 0, :] = 0.0  # lat row 0 dry at every retained cell/time -> 1.0
    pred[:, :, :, 0] = 0.0  # lon col 0 dry at every retained cell/time -> 1.0
    pred[:, 1, 1, 1] = 0.0  # cell (1, 1) dry at retained t=1 only -> 1/2
    pred[:, 0, 2, 2] = 0.0  # cell (2, 2) dry only at the dropped IC (t=0) -> 0
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        target={"PRATEsfc": torch.ones(SHAPE)},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = NearZeroFractionAggregator(
        area_weighted_mean=_ops().area_weighted_mean, include_maps=True
    )
    agg.record_batch(data)
    logs = agg.get_logs("val")
    # scalar metric still present
    assert logs["val/gen/PRATEsfc"] > 0.0
    # side-by-side comparison and error maps logged (as wandb Images)
    assert "val/gen_target_map/PRATEsfc" in logs
    assert "val/error_map/PRATEsfc" in logs
    assert "val/gen_map/PRATEsfc" not in logs  # only when target absent

    ds = agg.get_dataset()
    gen_map = ds["gen_map-PRATEsfc"].values
    target_map = ds["target_map-PRATEsfc"].values
    error_map = ds["error_map-PRATEsfc"].values
    # per-cell fraction of retained (sample, time) entries at/below 0
    assert (gen_map[0, :] == 1.0).all()  # fully-dry lat row
    assert (gen_map[:, 0] == 1.0).all()  # fully-dry lon col
    # cell (1, 1) is dry at 1 of 2 retained timesteps for both samples -> 1/2
    assert gen_map[1, 1] == pytest.approx(1.0 / 2.0)
    # cell (2, 2) is dry only at the dropped initial condition -> 0
    assert gen_map[2, 2] == 0.0
    assert (target_map == 0.0).all()
    assert (error_map == gen_map).all()


def test_near_zero_fraction_scalar_is_area_weighted():
    # Non-uniform (latitudinal) area weights: half the cells are dry but they
    # carry 3/(3+1) of the area, so the area-weighted fraction must be 0.75 --
    # a plain cell-count fraction would give 0.5.  (Weights must be
    # longitudinally uniform, as LatLonOperations assumes.)
    lat, lon = SHAPE[-2:]
    weights = torch.ones(lat, lon)
    weights[: lat // 2, :] = 3.0  # first lat-half carries 3x the area
    ops = LatLonOperations(area_weights=weights)
    pred = torch.ones(SHAPE)
    pred[:, :, : lat // 2, :] = 0.0  # only the heavily-weighted lat-half is dry
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = NearZeroFractionAggregator(area_weighted_mean=ops.area_weighted_mean)
    agg.record_batch(data)
    assert agg.get_logs("")["gen/PRATEsfc"] == pytest.approx(0.75)


def test_near_zero_fraction_drops_initial_condition_timestep():
    # dry only at the first timestep (of 3).
    pred = torch.ones(SHAPE)
    pred[:, 0] = 0.0
    time = InferenceBatchData.new_test_data().time
    # i_time_start == 0: that timestep is the initial condition and is dropped,
    # so none of the 2 retained timesteps are dry -> fraction 0.
    agg_ic = NearZeroFractionAggregator(area_weighted_mean=_ops().area_weighted_mean)
    agg_ic.record_batch(
        InferenceBatchData(prediction={"PRATEsfc": pred}, time=time, i_time_start=0)
    )
    assert agg_ic.get_logs("")["gen/PRATEsfc"] == 0.0
    # i_time_start > 0: no initial condition to drop, all 3 timesteps retained,
    # dry at 1 of 3 -> fraction 1/3.
    agg_mid = NearZeroFractionAggregator(area_weighted_mean=_ops().area_weighted_mean)
    agg_mid.record_batch(
        InferenceBatchData(prediction={"PRATEsfc": pred}, time=time, i_time_start=10)
    )
    assert agg_mid.get_logs("")["gen/PRATEsfc"] == pytest.approx(1.0 / 3.0)


def test_near_zero_fraction_maps_no_target():
    pred = torch.zeros(SHAPE)
    data = InferenceBatchData(
        prediction={"PRATEsfc": pred},
        time=InferenceBatchData.new_test_data().time,
        i_time_start=0,
    )
    agg = NearZeroFractionAggregator(
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


def test_near_zero_fraction_metric_config_include_maps_default_false():
    assert NearZeroFractionMetricConfig().include_maps is False


def test_near_zero_fraction_metric_config_disabled_by_default():
    config = NearZeroFractionMetricConfig()
    assert config.enabled is False
    assert config.get_name() == "near_zero_fraction"


def test_near_zero_fraction_metric_config_enabled_requires_variables():
    with pytest.raises(ValueError, match="no variables"):
        NearZeroFractionMetricConfig(enabled=True, threshold=1e-6)
    # enabled with variables and a positive threshold is fine
    NearZeroFractionMetricConfig(enabled=True, variables=["PRATEsfc"], threshold=1e-6)


def test_near_zero_fraction_metric_config_enabled_requires_positive_threshold():
    # the default threshold of 0.0 is rejected when enabled
    with pytest.raises(ValueError, match="threshold must be > 0"):
        NearZeroFractionMetricConfig(enabled=True, variables=["PRATEsfc"])
    with pytest.raises(ValueError, match="threshold must be > 0"):
        NearZeroFractionMetricConfig(
            enabled=True, variables=["PRATEsfc"], threshold=-1.0
        )
    # per-variable thresholds must also be positive
    with pytest.raises(ValueError, match="per_variable_threshold"):
        NearZeroFractionMetricConfig(
            enabled=True,
            variables=["PRATEsfc"],
            threshold=1e-6,
            per_variable_threshold={"PRATEsfc": 0.0},
        )
    # a disabled config with a non-positive threshold is allowed (no validation)
    NearZeroFractionMetricConfig(threshold=0.0)
