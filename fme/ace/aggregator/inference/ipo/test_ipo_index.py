import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from matplotlib import pyplot as plt

from fme import get_device
from fme.ace.aggregator.inference.data import InferenceBatchData

from ..utils import LatLonRegion
from .ipo_index import (
    MIN_YEARS_FOR_FILTERED_TPI,
    PairedIPOIndexAggregator,
    _IPORegionalAccumulator,
    low_pass_filter,
)


def _make_lat_lon():
    """Create lat/lon tensors spanning the TPI regions."""
    lat = torch.linspace(-60.0, 60.0, 61)
    lon = torch.linspace(100.0, 300.0, 81)
    return lat, lon


def _make_time(
    n_samples: int,
    n_times: int,
    start_time: tuple[int, ...] = (2000, 1, 1),
    freq: str = "6h",
    i_start: int = 0,
    calendar: str = "noleap",
) -> xr.DataArray:
    base = cftime.datetime(*start_time, calendar=calendar)
    all_times = xr.date_range(
        start=base, periods=i_start + n_times, freq=freq, use_cftime=True
    )
    time_values = all_times[i_start:].values
    return xr.concat(
        [xr.DataArray(time_values, dims=("time",)) for _ in range(n_samples)],
        dim="sample",
    )


def _make_sst_data(
    n_samples: int,
    n_times: int,
    n_lat: int,
    n_lon: int,
    sst_name: str = "sst",
    constant_value: float = 300.0,
) -> dict[str, torch.Tensor]:
    sst = torch.full(
        (n_samples, n_times, n_lat, n_lon),
        constant_value,
        device=get_device(),
    )
    return {sst_name: sst}


class TestLowPassFilter:
    def test_constant_signal_unchanged(self):
        data = np.ones(500)
        filtered = low_pass_filter(data)
        np.testing.assert_allclose(filtered, data, atol=1e-6)

    def test_high_frequency_removed(self):
        n_months = 600
        t = np.arange(n_months)
        high_freq = np.sin(2 * np.pi * t / 6)  # 6-month period
        low_freq = np.sin(2 * np.pi * t / (20 * 12))  # 20-year period
        combined = high_freq + low_freq

        filtered = low_pass_filter(combined, cutoff_period_yrs=13.0)
        # Trim edges to avoid filter transient effects
        trim = 13 * 12
        residual_high_freq = filtered[trim:-trim] - low_freq[trim:-trim]
        assert np.std(residual_high_freq) < 0.1 * np.std(high_freq)

    def test_low_frequency_preserved(self):
        n_months = 600
        t = np.arange(n_months)
        low_freq = np.sin(2 * np.pi * t / (20 * 12))  # 20-year period

        filtered = low_pass_filter(low_freq, cutoff_period_yrs=13.0)
        correlation = np.corrcoef(filtered, low_freq)[0, 1]
        assert correlation > 0.95


class TestIPORegionalAccumulator:
    def test_accumulates_batches(self):
        lat, lon = _make_lat_lon()
        n_samples, n_times, n_lat, n_lon = 2, 20, len(lat), len(lon)

        regions = {
            name: LatLonRegion(
                lat=lat,
                lon=lon,
                lat_bounds=spec["lat_bounds"],
                lon_bounds=spec["lon_bounds"],
            )
            for name, spec in [
                ("T1", {"lat_bounds": (25.0, 45.0), "lon_bounds": (140.0, 215.0)}),
                ("T2", {"lat_bounds": (-10.0, 10.0), "lon_bounds": (170.0, 270.0)}),
                ("T3", {"lat_bounds": (-50.0, -15.0), "lon_bounds": (150.0, 200.0)}),
            ]
        }

        accumulator = _IPORegionalAccumulator(regions)

        time1 = _make_time(n_samples, n_times // 2)
        data1 = _make_sst_data(n_samples, n_times // 2, n_lat, n_lon, sst_name="sst")
        batch1 = InferenceBatchData(
            prediction=data1,
            time=time1,
            i_time_start=0,
        )
        accumulator.record_batch(batch1)

        time2 = _make_time(n_samples, n_times // 2, i_start=n_times // 2)
        data2 = _make_sst_data(n_samples, n_times // 2, n_lat, n_lon, sst_name="sst")
        batch2 = InferenceBatchData(
            prediction=data2,
            time=time2,
            i_time_start=n_times // 2,
        )
        accumulator.record_batch(batch2)

        assert accumulator._raw_times is not None
        assert accumulator._raw_times.sizes["time"] == n_times
        for region_name in regions:
            assert "sst" in accumulator._raw_means[region_name]
            assert accumulator._raw_means[region_name]["sst"].shape[1] == n_times

    def test_constant_sst_gives_zero_tpi(self):
        """Constant SST should give zero anomalies and hence zero TPI."""
        lat, lon = _make_lat_lon()
        n_lat, n_lon = len(lat), len(lon)
        n_samples = 1
        n_months = 48
        steps_per_month = 120  # 6h steps in ~30 days
        n_times = n_months * steps_per_month

        regions = {
            name: LatLonRegion(
                lat=lat,
                lon=lon,
                lat_bounds=spec["lat_bounds"],
                lon_bounds=spec["lon_bounds"],
            )
            for name, spec in [
                ("T1", {"lat_bounds": (25.0, 45.0), "lon_bounds": (140.0, 215.0)}),
                ("T2", {"lat_bounds": (-10.0, 10.0), "lon_bounds": (170.0, 270.0)}),
                ("T3", {"lat_bounds": (-50.0, -15.0), "lon_bounds": (150.0, 200.0)}),
            ]
        }

        accumulator = _IPORegionalAccumulator(regions)

        time = _make_time(n_samples, n_times)
        data = _make_sst_data(
            n_samples, n_times, n_lat, n_lon, sst_name="sst", constant_value=300.0
        )
        batch = InferenceBatchData(prediction=data, time=time, i_time_start=0)
        accumulator.record_batch(batch)

        tpi_ds = accumulator.get_tpi_indices()
        if "sst" in tpi_ds:
            tpi_values = tpi_ds["sst"].values
            np.testing.assert_allclose(
                tpi_values[~np.isnan(tpi_values)], 0.0, atol=5e-5
            )


class TestPairedIPOIndexAggregator:
    @pytest.mark.medium
    def test_get_logs_returns_expected_keys(self):
        """Test that get_logs returns the expected metric keys for long runs."""
        lat = torch.linspace(-60.0, 60.0, 13)
        lon = torch.linspace(100.0, 300.0, 17)
        n_lat, n_lon = len(lat), len(lon)
        n_samples = 1
        n_months = (MIN_YEARS_FOR_FILTERED_TPI + 5) * 12

        agg = PairedIPOIndexAggregator(lat=lat, lon=lon)

        chunk_size = 60
        for i_start in range(0, n_months, chunk_size):
            n_chunk = min(chunk_size, n_months - i_start)
            time = _make_time(n_samples, n_chunk, i_start=i_start, freq="MS")
            months = time.isel(sample=0).dt.month.values
            seasonal = torch.tensor(
                [np.sin(2 * np.pi * m / 12) for m in months],
                device=get_device(),
                dtype=torch.float32,
            )[None, :, None, None]
            target_sst = (
                torch.full(
                    (n_samples, n_chunk, n_lat, n_lon), 300.0, device=get_device()
                )
                + seasonal
            )
            pred_sst = (
                torch.full(
                    (n_samples, n_chunk, n_lat, n_lon), 300.5, device=get_device()
                )
                + seasonal * 1.2
            )

            batch = InferenceBatchData(
                prediction={"sst": pred_sst},
                target={"sst": target_sst},
                time=time,
                i_time_start=i_start,
            )
            agg.record_batch(batch)

        logs = agg.get_logs("test")
        plt.close("all")

        assert any("ipo_tpi_std_norm" in k for k in logs)
        assert any("ipo_tpi_std" in k for k in logs)
        assert any("ipo_tpi_filtered" in k for k in logs)
        assert any("ipo_tpi_power_spectrum" in k for k in logs)
        assert not any("ipo_tpi_std_ratio" in k for k in logs)

    @pytest.mark.medium
    def test_get_logs_empty_for_short_rollout(self):
        """Rollouts shorter than the minimum should not report IPO metrics."""
        lat = torch.linspace(-60.0, 60.0, 13)
        lon = torch.linspace(100.0, 300.0, 17)
        n_lat, n_lon = len(lat), len(lon)
        n_samples = 1
        n_months = (MIN_YEARS_FOR_FILTERED_TPI // 2) * 12

        agg = PairedIPOIndexAggregator(lat=lat, lon=lon)

        chunk_size = 60
        for i_start in range(0, n_months, chunk_size):
            n_chunk = min(chunk_size, n_months - i_start)
            time = _make_time(n_samples, n_chunk, i_start=i_start, freq="MS")
            months = time.isel(sample=0).dt.month.values
            seasonal = torch.tensor(
                [np.sin(2 * np.pi * m / 12) for m in months],
                device=get_device(),
                dtype=torch.float32,
            )[None, :, None, None]
            sst = (
                torch.full(
                    (n_samples, n_chunk, n_lat, n_lon), 300.0, device=get_device()
                )
                + seasonal
            )
            batch = InferenceBatchData(
                prediction={"sst": sst},
                target={"sst": sst},
                time=time,
                i_time_start=i_start,
            )
            agg.record_batch(batch)

        logs = agg.get_logs("test")
        plt.close("all")

        assert not any("ipo_tpi_std_norm" in k for k in logs)
        assert not any("ipo_tpi_std" in k for k in logs)
        assert not any("ipo_tpi_filtered" in k for k in logs)
        assert not any("ipo_tpi_power_spectrum" in k for k in logs)

    def test_get_dataset_has_sources(self):
        """Test that get_dataset returns target and prediction sources."""
        lat, lon = _make_lat_lon()
        n_lat, n_lon = len(lat), len(lon)
        n_samples = 1
        n_months = 36
        steps_per_month = 120
        n_times = n_months * steps_per_month

        agg = PairedIPOIndexAggregator(lat=lat, lon=lon)

        time = _make_time(n_samples, n_times)
        sst = torch.randn(n_samples, n_times, n_lat, n_lon, device=get_device()) + 300.0
        batch = InferenceBatchData(
            prediction={"sst": sst},
            target={"sst": sst + 0.1},
            time=time,
            i_time_start=0,
        )
        agg.record_batch(batch)

        ds = agg.get_dataset()
        if len(ds) > 0:
            assert "source" in ds.dims
            assert "target" in ds.source.values
            assert "prediction" in ds.source.values
