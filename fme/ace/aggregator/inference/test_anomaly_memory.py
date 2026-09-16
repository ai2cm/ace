import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.coordinates import HEALPixCoordinates, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.testing import mock_distributed

from .anomaly_memory import (
    AnomalyMemoryAggregator,
    AnomalyMemoryMetricConfig,
    LaggedAnomalyMoments,
    annual_harmonic_basis,
    month_mask,
)
from .build_context import MetricBuildContext, MetricNotSupportedError, maybe_filter
from .data import InferenceBatchData
from .main import InferenceEvaluatorAggregatorConfig
from .utils import LatLonBoxConfig

LAT = torch.tensor([-60.0, -20.0, 20.0, 60.0])
LON = torch.tensor([0.0, 90.0, 180.0, 270.0])
LAGS = [0, 1, 2, 5, 10]
TIMESTEP = datetime.timedelta(days=1)


def _daily_time(n_sample: int, n_time: int, calendar: str = "noleap") -> xr.DataArray:
    start = cftime.datetime(2000, 1, 1, calendar=calendar)
    times = [start + i * TIMESTEP for i in range(n_time)]
    return xr.DataArray([times for _ in range(n_sample)], dims=["sample", "time"])


def _seasonal_ar1(
    time: xr.DataArray,
    rho: float,
    seed: int,
    amplitude: float = 10.0,
    noise: float = 1.0,
) -> torch.Tensor:
    """AR(1) noise on top of an annual cycle, shape (sample, time, lat, lon)."""
    rng = np.random.default_rng(seed)
    n_sample, n_time = time.shape
    doy = time.dt.dayofyear.values
    cycle = amplitude * np.cos(2 * np.pi * doy / 365.0)
    x = np.zeros((n_sample, n_time, len(LAT), len(LON)))
    for t in range(n_time):
        eps = rng.standard_normal((n_sample, len(LAT), len(LON))) * noise
        x[:, t] = eps if t == 0 else rho * x[:, t - 1] + eps
    x += cycle[:, :, None, None]
    return torch.tensor(x, dtype=torch.float32, device=get_device())


def _batch(
    time: xr.DataArray,
    fields: dict[str, torch.Tensor],
    i_time_start: int,
    target: dict[str, torch.Tensor] | None = None,
) -> InferenceBatchData:
    target = fields if target is None else target
    return InferenceBatchData(
        prediction=fields, target=target, time=time, i_time_start=i_time_start
    )


def _aggregator(**kwargs) -> AnomalyMemoryAggregator:
    defaults = dict(
        lat=LAT,
        lon=LON,
        lags=LAGS,
        report_lags=[5],
        map_lags=[5],
        n_harmonics=2,
        months_northern=list(range(1, 13)),
        months_southern=list(range(1, 13)),
        regions=[LatLonBoxConfig("north", [0.0, 90.0], [0.0, 360.0])],
    )
    defaults.update(kwargs)
    return AnomalyMemoryAggregator(**defaults)


def brute_force_lagged_cov(
    x: np.ndarray, basis: np.ndarray, lags: list[int], season: np.ndarray
) -> np.ndarray:
    """Two-pass reference: fit the climatology, subtract, correlate.

    x is (sample, time, *spatial); the climatology is pooled over samples.
    """
    n_sample, n_time = x.shape[:2]
    flat = x.reshape(n_sample * n_time, -1)
    design = basis.reshape(n_sample * n_time, -1)
    coeffs = np.linalg.lstsq(design, flat, rcond=None)[0]
    anomaly = (flat - design @ coeffs).reshape(x.shape)
    out = []
    for lag in lags:
        lead = season.copy()
        if lag > 0:
            lead[:, n_time - lag :] = False
        products = [
            anomaly[s, t] * anomaly[s, t + lag]
            for s in range(n_sample)
            for t in np.flatnonzero(lead[s])
        ]
        out.append(np.mean(products, axis=0))
    return np.stack(out)


def _stream(
    moments: LaggedAnomalyMoments,
    x: torch.Tensor,
    basis: torch.Tensor,
    season: torch.Tensor,
    window: int,
) -> None:
    n_time = x.shape[1]
    for start in range(0, n_time, window):
        stop = min(start + window, n_time)
        moments.update(
            x[:, start:stop],
            basis[:, start:stop],
            season[:, start:stop],
            new_trajectories=start == 0,
        )


def _finalize(moments: LaggedAnomalyMoments) -> np.ndarray:
    from fme.core.distributed import Distributed

    state = moments.reduced_state(Distributed.get_instance())
    return LaggedAnomalyMoments.finalize(state).cpu().numpy()


@pytest.mark.parametrize("window", [7, 25, 400])
def test_streaming_matches_brute_force(window: int):
    time = _daily_time(n_sample=2, n_time=400)
    x = _seasonal_ar1(time, rho=0.8, seed=0)
    basis = annual_harmonic_basis(time, n_harmonics=2)
    season = month_mask(time, [1, 2, 3, 10, 11, 12])
    moments = LaggedAnomalyMoments(LAGS, basis.shape[-1], x.shape[2:])
    _stream(moments, x, basis, season, window)
    streamed = _finalize(moments)
    expected = brute_force_lagged_cov(
        x.cpu().numpy().astype(np.float64),
        basis.cpu().numpy(),
        LAGS,
        season.cpu().numpy(),
    )
    np.testing.assert_allclose(streamed, expected, rtol=1e-8, atol=1e-8)


def test_ar1_correlation_recovered():
    rho = 0.85
    time = _daily_time(n_sample=4, n_time=365 * 8)
    x = _seasonal_ar1(time, rho=rho, seed=1)
    basis = annual_harmonic_basis(time, n_harmonics=2)
    season = torch.ones(time.shape, dtype=torch.bool, device=get_device())
    moments = LaggedAnomalyMoments(LAGS, basis.shape[-1], x.shape[2:])
    _stream(moments, x, basis, season, window=50)
    cov = _finalize(moments)
    corr = cov / cov[0]
    for i, lag in enumerate(LAGS):
        assert corr[i].mean() == pytest.approx(rho**lag, abs=0.03)


def test_pure_seasonal_cycle_is_masked():
    time = _daily_time(n_sample=1, n_time=365 * 3)
    x = _seasonal_ar1(time, rho=0.0, seed=2, amplitude=50.0, noise=0.0)
    basis = annual_harmonic_basis(time, n_harmonics=1)
    season = torch.ones(time.shape, dtype=torch.bool, device=get_device())
    moments = LaggedAnomalyMoments(LAGS, basis.shape[-1], x.shape[2:])
    _stream(moments, x, basis, season, window=100)
    cov = _finalize(moments)
    assert np.isnan(cov).all()


def test_constant_cells_are_masked():
    """A cell that never varies has no anomaly variance; roundoff in the
    streaming algebra must not turn it into a spurious correlation."""
    time = _daily_time(n_sample=2, n_time=300)
    x = _seasonal_ar1(time, rho=0.7, seed=14)
    x[:, :, 2, 3] = 0.37
    x[:, :, 3, 3] = 0.0
    basis = annual_harmonic_basis(time, n_harmonics=2)
    season = torch.ones(time.shape, dtype=torch.bool, device=get_device())
    moments = LaggedAnomalyMoments(LAGS, basis.shape[-1], x.shape[2:])
    _stream(moments, x, basis, season, window=40)
    cov = _finalize(moments)
    assert np.isnan(cov[:, 2, 3]).all()
    assert np.isnan(cov[:, 3, 3]).all()
    assert np.isfinite(cov[:, 0, 0]).all()


def test_nan_cells_masked_and_finite_cells_unaffected():
    time = _daily_time(n_sample=2, n_time=200)
    x = _seasonal_ar1(time, rho=0.7, seed=3)
    basis = annual_harmonic_basis(time, n_harmonics=2)
    season = torch.ones(time.shape, dtype=torch.bool, device=get_device())
    clean = LaggedAnomalyMoments(LAGS, basis.shape[-1], x.shape[2:])
    _stream(clean, x, basis, season, window=30)
    masked_x = x.clone()
    masked_x[:, :, 0, 0] = float("nan")
    masked_x[0, 17, 1, 1] = float("nan")
    masked = LaggedAnomalyMoments(LAGS, basis.shape[-1], x.shape[2:])
    _stream(masked, masked_x, basis, season, window=30)
    clean_cov = _finalize(clean)
    masked_cov = _finalize(masked)
    assert np.isnan(masked_cov[:, 0, 0]).all()
    assert np.isnan(masked_cov[:, 1, 1]).all()
    others = np.ones(x.shape[2:], dtype=bool)
    others[0, 0] = others[1, 1] = False
    np.testing.assert_allclose(masked_cov[:, others], clean_cov[:, others])


def test_hemisphere_months_restrict_pairs():
    """Scoring only northern winter must match a brute force restricted to
    those months in the northern rows, while the south uses its own window."""
    time = _daily_time(n_sample=1, n_time=365 * 2)
    x = _seasonal_ar1(time, rho=0.6, seed=4)
    agg = _aggregator(months_northern=[12, 1, 2], months_southern=[6, 7, 8])
    agg.record_batch(_batch(time, {"a": x}, i_time_start=1))
    corr = agg.get_dataset()["corr-a"].sel(source="prediction").values
    basis = annual_harmonic_basis(time, n_harmonics=2).cpu().numpy()
    x_np = x.cpu().numpy().astype(np.float64)
    for rows, months in (([2, 3], [12, 1, 2]), ([0, 1], [6, 7, 8])):
        season = np.isin(time.dt.month.values, months)
        expected = brute_force_lagged_cov(x_np[:, :, rows], basis, LAGS, season)
        np.testing.assert_allclose(
            corr[:, rows], expected / expected[0], rtol=1e-6, atol=1e-6
        )


def test_streaming_matches_single_batch_through_aggregator():
    time = _daily_time(n_sample=2, n_time=120)
    x = _seasonal_ar1(time, rho=0.8, seed=5)
    y = _seasonal_ar1(time, rho=0.3, seed=6)
    single = _aggregator()
    single.record_batch(_batch(time, {"a": x}, i_time_start=0, target={"a": y}))
    streamed = _aggregator()
    for start in range(0, 120, 25):
        stop = min(start + 25, 120)
        streamed.record_batch(
            _batch(
                time.isel(time=slice(start, stop)),
                {"a": x[:, start:stop]},
                i_time_start=start,
                target={"a": y[:, start:stop]},
            )
        )
    xr.testing.assert_allclose(single.get_dataset(), streamed.get_dataset())


def test_initial_condition_dropped_from_first_window():
    time = _daily_time(n_sample=1, n_time=100)
    x = _seasonal_ar1(time, rho=0.8, seed=7)
    with_ic = _aggregator()
    with_ic.record_batch(_batch(time, {"a": x}, i_time_start=0))
    without_ic = _aggregator()
    without_ic.record_batch(
        _batch(time.isel(time=slice(1, None)), {"a": x[:, 1:]}, i_time_start=1)
    )
    xr.testing.assert_allclose(with_ic.get_dataset(), without_ic.get_dataset())


def test_logs_and_dataset_structure():
    time = _daily_time(n_sample=2, n_time=200)
    x = _seasonal_ar1(time, rho=0.9, seed=8)
    y = _seasonal_ar1(time, rho=0.2, seed=9)
    agg = _aggregator()
    agg.record_batch(_batch(time, {"a": x}, i_time_start=0, target={"a": y}))
    logs = agg.get_logs(label="anomaly_memory")
    assert "anomaly_memory/maps/a-lag5" in logs
    gen = logs["anomaly_memory/prediction/a-north-lag5"]
    target = logs["anomaly_memory/target/a-north-lag5"]
    assert gen > target
    assert logs["anomaly_memory/gap/a-north-lag5"] == pytest.approx(gen - target)
    assert logs["anomaly_memory/variance_ratio/a-north"] > 1.0
    ds = agg.get_dataset()
    assert ds["corr-a"].dims == ("source", "lag", "lat", "lon")
    assert ds["variance-a"].dims == ("source", "lat", "lon")
    assert ds["region_corr-a"].dims == ("source", "region", "lag")
    np.testing.assert_allclose(ds["corr-a"].sel(lag=0).values, 1.0)
    assert ds["region_corr-a"].sel(
        source="prediction", region="north", lag=5
    ).item() == pytest.approx(gen)


def test_no_regions_gives_empty_region_axis():
    time = _daily_time(n_sample=1, n_time=60)
    x = _seasonal_ar1(time, rho=0.5, seed=13)
    agg = _aggregator(regions=[])
    agg.record_batch(_batch(time, {"a": x}, i_time_start=0))
    ds = agg.get_dataset()
    assert ds["region_corr-a"].shape == (2, 0, len(LAGS))
    assert not any(key.startswith("prediction/") for key in agg.get_logs(label=""))


def test_maps_disabled_with_empty_map_lags():
    time = _daily_time(n_sample=1, n_time=60)
    x = _seasonal_ar1(time, rho=0.5, seed=10)
    agg = _aggregator(map_lags=[])
    agg.record_batch(_batch(time, {"a": x}, i_time_start=0))
    assert not any("maps/" in key for key in agg.get_logs(label=""))


def test_metrics_call_distributed():
    time = _daily_time(n_sample=1, n_time=60)
    x = _seasonal_ar1(time, rho=0.5, seed=11)
    with mock_distributed(0.0) as mock:
        agg = _aggregator()
        agg.record_batch(_batch(time, {"a": x}, i_time_start=0))
        agg.get_dataset()
        assert mock.reduce_called


def test_variable_filtering():
    time = _daily_time(n_sample=1, n_time=60)
    x = _seasonal_ar1(time, rho=0.5, seed=12)
    agg = maybe_filter(_aggregator(), ["a"])
    agg.record_batch(_batch(time, {"a": x, "b": x}, i_time_start=0))
    assert sorted(agg.get_dataset().data_vars) == [
        "corr-a",
        "region_corr-a",
        "variance-a",
    ]


def test_config_validation():
    with pytest.raises(ValueError, match="include 0"):
        AnomalyMemoryMetricConfig(lags=[1, 7])
    with pytest.raises(ValueError, match="report_lags"):
        AnomalyMemoryMetricConfig(lags=[0, 7], report_lags=[14])
    with pytest.raises(ValueError, match="map_lags"):
        AnomalyMemoryMetricConfig(lags=[0, 7], map_lags=[3])
    with pytest.raises(ValueError, match="months_northern"):
        AnomalyMemoryMetricConfig(months_northern=[13])
    with pytest.raises(ValueError, match="unique"):
        AnomalyMemoryMetricConfig(
            regions=[
                LatLonBoxConfig("r", [0, 10], [0, 10]),
                LatLonBoxConfig("r", [0, 10], [0, 10]),
            ]
        )
    with pytest.raises(ValueError, match="south, north"):
        LatLonBoxConfig("r", [10, 0], [0, 10])
    with pytest.raises(ValueError, match="2 values"):
        LatLonBoxConfig("r", [0, 10, 20], [0, 10])


def _build_context(coords, n_forward_steps: int) -> MetricBuildContext:
    ds_info = DatasetInfo(horizontal_coordinates=coords, timestep=TIMESTEP)
    return MetricBuildContext(
        ops=ds_info.gridded_operations,
        horizontal_coordinates=coords,
        n_timesteps=n_forward_steps + 1,
        n_ic_steps=1,
        timestep=TIMESTEP,
        variable_metadata=None,
        channel_mean_names=None,
        monthly_reference_data=None,
        time_mean_reference_data=None,
        initial_time=_daily_time(1, 1).isel(time=slice(0, 0)),
    )


def test_build_rejects_healpix_and_short_runs():
    config = AnomalyMemoryMetricConfig(enabled=True)
    healpix = HEALPixCoordinates(
        face=torch.arange(12), height=torch.arange(4), width=torch.arange(4)
    )
    with pytest.raises(MetricNotSupportedError, match="LatLonCoordinates"):
        config.build(_build_context(healpix, n_forward_steps=100))
    latlon = LatLonCoordinates(lat=LAT, lon=LON)
    with pytest.raises(MetricNotSupportedError, match="forward steps"):
        config.build(_build_context(latlon, n_forward_steps=30))
    assert config.build(_build_context(latlon, n_forward_steps=31)).aggregator


def test_metric_config_disabled_by_default_and_wired():
    assert AnomalyMemoryMetricConfig().enabled is False
    names = [m.get_name() for m in InferenceEvaluatorAggregatorConfig()._get_metrics()]
    assert "anomaly_memory" not in names
    config = InferenceEvaluatorAggregatorConfig(
        anomaly_memory=AnomalyMemoryMetricConfig(enabled=True, variables=["a"])
    )
    assert "anomaly_memory" in [m.get_name() for m in config._get_metrics()]
