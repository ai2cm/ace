import math

import numpy as np
import pytest
import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.metrics import spherical_area_weights

from .data import InferenceBatchData, make_dummy_time
from .main import InferenceEvaluatorAggregatorConfig
from .wind_consistency import (
    WindConsistencyAggregator,
    WindConsistencyMetricConfig,
    interpolate_to_pressure_log_linear,
)

N_LAT, N_LON = 3, 4
N_LEVEL = 8


def _coords() -> LatLonCoordinates:
    return LatLonCoordinates(
        lon=torch.linspace(0.0, 360.0, N_LON + 1)[:-1],
        lat=torch.linspace(-60.0, 60.0, N_LAT),
    )


def _ops() -> LatLonOperations:
    return LatLonOperations(_coords().area_weights.to(get_device()))


def _vertical_coordinate() -> HybridSigmaPressureCoordinate:
    """Pure-sigma coordinate: interface pressures = bk * ps, increasing in index.

    With N_LEVEL midpoints we need N_LEVEL + 1 interface coefficients.
    """
    bk = torch.linspace(0.0, 1.0, N_LEVEL + 1)
    ak = torch.zeros(N_LEVEL + 1)
    return HybridSigmaPressureCoordinate(ak=ak, bk=bk)


def test_interpolate_log_linear_recovers_analytic_value():
    # A column with known monotonic pressures and a field that is linear in
    # log(p) must be recovered exactly at an in-range target.
    pressures = torch.tensor([1e4, 2e4, 4e4, 8e4])  # Pa, strictly increasing
    log_p = torch.log(pressures)
    # field = 3 + 2 * log(p): linear in log(p)
    field = 3.0 + 2.0 * log_p
    target_pa = 3e4
    result = interpolate_to_pressure_log_linear(field, pressures, target_pa)
    expected = 3.0 + 2.0 * math.log(target_pa)
    np.testing.assert_allclose(result.item(), expected, rtol=1e-5)


def test_interpolate_log_linear_out_of_range_is_nan():
    pressures = torch.tensor([1e4, 2e4, 4e4, 8e4])
    field = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # below the bottom level
    assert torch.isnan(interpolate_to_pressure_log_linear(field, pressures, 9e4))
    # above the top level
    assert torch.isnan(interpolate_to_pressure_log_linear(field, pressures, 5e3))


def _batch(
    *,
    u_surface: torch.Tensor,
    v_surface: torch.Tensor,
    u_levels: torch.Tensor,
    v_levels: torch.Tensor,
    surface_pressure: torch.Tensor,
    surface_hpa: int = 500,
    target: bool = False,
    i_time_start: int = 1,
) -> InferenceBatchData:
    """Build an InferenceBatchData with the named winds and surface pressure.

    ``*_levels`` have shape (sample, time, lat, lon, n_level); the others have
    shape (sample, time, lat, lon).
    """
    n_sample, n_time = u_surface.shape[:2]
    time = make_dummy_time(n_sample, n_time)
    n_level = u_levels.shape[-1]
    stream = {
        f"UGRD{surface_hpa}": u_surface,
        f"VGRD{surface_hpa}": v_surface,
        "PRESsfc": surface_pressure,
    }
    for i in range(n_level):
        stream[f"eastward_wind_{i}"] = u_levels[..., i].contiguous()
        stream[f"northward_wind_{i}"] = v_levels[..., i].contiguous()
    return InferenceBatchData(
        prediction=stream,
        prediction_norm=stream,
        target=stream if target else None,
        target_norm=stream if target else None,
        time=time,
        i_time_start=i_time_start,
    )


def _consistent_inputs(n_sample=2, n_time=4, ps=1e5, surface_hpa=500):
    """Build inputs where the surface winds equal the model-level interpolant.

    The model-level winds are linear in log(p); the surface winds are set to
    the analytic log-linear interpolant at ``surface_hpa``, so the residuals
    are exactly zero.
    """
    vc = _vertical_coordinate()
    surface_pressure = torch.full((n_sample, n_time, N_LAT, N_LON), float(ps))
    interfaces = vc.interface_pressure(surface_pressure)
    midpoints = 0.5 * (interfaces[..., :-1] + interfaces[..., 1:])
    log_mid = torch.log(midpoints)
    # u = 1 + 2 log(p), v = -3 + 0.5 log(p): both linear in log(p)
    u_levels = 1.0 + 2.0 * log_mid
    v_levels = -3.0 + 0.5 * log_mid
    target_pa = surface_hpa * 100.0
    u_surface = 1.0 + 2.0 * math.log(target_pa)
    v_surface = -3.0 + 0.5 * math.log(target_pa)
    u_surface = torch.full((n_sample, n_time, N_LAT, N_LON), float(u_surface))
    v_surface = torch.full((n_sample, n_time, N_LAT, N_LON), float(v_surface))
    return vc, u_surface, v_surface, u_levels, v_levels, surface_pressure


def test_aggregator_logs_have_expected_keys_and_zero_residual():
    vc, u_surface, v_surface, u_levels, v_levels, ps = _consistent_inputs()
    agg = WindConsistencyAggregator(
        gridded_operations=_ops(),
        horizontal_coordinates=_coords(),
        vertical_coordinate=vc,
        surfaces_hpa=[500],
    )
    agg.record_batch(
        _batch(
            u_surface=u_surface,
            v_surface=v_surface,
            u_levels=u_levels,
            v_levels=v_levels,
            surface_pressure=ps,
            target=True,
            i_time_start=1,
        )
    )
    logs = agg.get_logs("wind_consistency")
    for quantity in (
        "eastward_residual",
        "northward_residual",
        "residual_speed",
        "speed_residual",
    ):
        for stat in ("gen", "target", "bias", "rmse"):
            assert f"wind_consistency/{quantity}_500/{stat}" in logs
    # Surface winds equal the interpolant -> residual means ~0.
    for quantity in (
        "eastward_residual",
        "northward_residual",
        "residual_speed",
        "speed_residual",
    ):
        assert logs[f"wind_consistency/{quantity}_500/gen"] == pytest.approx(
            0.0, abs=1e-5
        )
        assert logs[f"wind_consistency/{quantity}_500/target"] == pytest.approx(
            0.0, abs=1e-5
        )
        assert logs[f"wind_consistency/{quantity}_500/rmse"] == pytest.approx(
            0.0, abs=1e-5
        )


def test_aggregator_no_target_leaves_target_and_rmse_nan():
    vc, u_surface, v_surface, u_levels, v_levels, ps = _consistent_inputs()
    # Perturb surface winds so the gen residual is nonzero and finite.
    agg = WindConsistencyAggregator(
        gridded_operations=_ops(),
        horizontal_coordinates=_coords(),
        vertical_coordinate=vc,
        surfaces_hpa=[500],
    )
    agg.record_batch(
        _batch(
            u_surface=u_surface + 1.0,
            v_surface=v_surface,
            u_levels=u_levels,
            v_levels=v_levels,
            surface_pressure=ps,
            target=False,
            i_time_start=1,
        )
    )
    logs = agg.get_logs("wind_consistency")
    assert logs["wind_consistency/eastward_residual_500/gen"] == pytest.approx(
        1.0, abs=1e-5
    )
    assert math.isnan(logs["wind_consistency/eastward_residual_500/target"])
    assert math.isnan(logs["wind_consistency/eastward_residual_500/bias"])
    assert math.isnan(logs["wind_consistency/eastward_residual_500/rmse"])


def test_aggregator_streaming_matches_single_batch():
    vc, u_surface, v_surface, u_levels, v_levels, ps = _consistent_inputs(
        n_sample=2, n_time=6
    )
    # Perturb so the residuals are nonzero (a more demanding equality check).
    u_surface = u_surface + 0.5 * torch.randn_like(u_surface)
    v_surface = v_surface + 0.5 * torch.randn_like(v_surface)

    def build_agg() -> WindConsistencyAggregator:
        return WindConsistencyAggregator(
            gridded_operations=_ops(),
            horizontal_coordinates=_coords(),
            vertical_coordinate=vc,
            surfaces_hpa=[500],
        )

    # i_time_start != 0 so no IC is skipped, so the split is exact.
    single = build_agg()
    single.record_batch(
        _batch(
            u_surface=u_surface,
            v_surface=v_surface,
            u_levels=u_levels,
            v_levels=v_levels,
            surface_pressure=ps,
            target=True,
            i_time_start=1,
        )
    )
    single_logs = single.get_logs("wc")

    streamed = build_agg()
    sl = [slice(0, 3), slice(3, 6)]
    for s in sl:
        streamed.record_batch(
            _batch(
                u_surface=u_surface[:, s],
                v_surface=v_surface[:, s],
                u_levels=u_levels[:, s],
                v_levels=v_levels[:, s],
                surface_pressure=ps[:, s],
                target=True,
                i_time_start=1,
            )
        )
    streamed_logs = streamed.get_logs("wc")

    assert single_logs.keys() == streamed_logs.keys()
    for key in single_logs:
        a, b = single_logs[key], streamed_logs[key]
        if math.isnan(a):
            assert math.isnan(b)
        else:
            assert a == pytest.approx(b, rel=1e-6, abs=1e-9)


def test_aggregator_get_dataset_holds_means():
    vc, u_surface, v_surface, u_levels, v_levels, ps = _consistent_inputs()
    agg = WindConsistencyAggregator(
        gridded_operations=_ops(),
        horizontal_coordinates=_coords(),
        vertical_coordinate=vc,
        surfaces_hpa=[500],
    )
    agg.record_batch(
        _batch(
            u_surface=u_surface,
            v_surface=v_surface,
            u_levels=u_levels,
            v_levels=v_levels,
            surface_pressure=ps,
            target=True,
            i_time_start=1,
        )
    )
    ds = agg.get_dataset()
    assert "eastward_residual_500" in ds
    assert list(ds["eastward_residual_500"].coords["source"].values) == [
        "target",
        "prediction",
    ]
    np.testing.assert_allclose(
        ds["eastward_residual_500"].values, [0.0, 0.0], atol=1e-5
    )


def test_wind_consistency_metric_config_disabled_by_default():
    assert WindConsistencyMetricConfig().enabled is False


def test_wind_consistency_excluded_from_default_metrics():
    names = [m.get_name() for m in InferenceEvaluatorAggregatorConfig()._get_metrics()]
    assert "wind_consistency" not in names


def test_wind_consistency_included_when_enabled():
    config = InferenceEvaluatorAggregatorConfig(
        wind_consistency=WindConsistencyMetricConfig(enabled=True)
    )
    names = [m.get_name() for m in config._get_metrics()]
    assert "wind_consistency" in names


def test_area_weights_match_spherical_area_weights():
    # Sanity: the aggregator builds the same weights helper used elsewhere.
    coords = _coords()
    w = spherical_area_weights(coords.lat_1d, N_LON)
    assert w.shape == (N_LAT, N_LON)
