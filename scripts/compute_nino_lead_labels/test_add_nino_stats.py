"""Tests for add_nino_stats core logic."""

import numpy as np
import xarray as xr
from add_nino_stats import compute_lead_stats


def test_compute_lead_stats_matches_numpy():
    time = xr.date_range(
        "2000-01-01", periods=10, freq="MS", calendar="noleap", use_cftime=True
    )
    vals = np.arange(10, dtype="float32")
    field = np.broadcast_to(vals[:, None, None], (10, 2, 3)).copy()
    ds = xr.Dataset(
        {"nino34_lead_01": (["time", "lat", "lon"], field)},
        coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
    )
    s = compute_lead_stats(ds)["nino34_lead_01"]
    np.testing.assert_allclose(s["mean"], float(vals.mean()), atol=1e-5)
    np.testing.assert_allclose(s["std"], float(vals.std()), atol=1e-5)  # ddof=0
    np.testing.assert_allclose(s["residual_std"], float(np.diff(vals).std()), atol=1e-5)


def test_compute_lead_stats_ignores_nan_tail():
    time = xr.date_range(
        "2000-01-01", periods=6, freq="MS", calendar="noleap", use_cftime=True
    )
    vals = np.array([1.0, 3.0, 5.0, np.nan, np.nan, np.nan], dtype="float32")
    field = np.broadcast_to(vals[:, None, None], (6, 1, 1)).copy()
    ds = xr.Dataset(
        {"nino34_lead_02": (["time", "lat", "lon"], field)},
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
    )
    s = compute_lead_stats(ds)["nino34_lead_02"]
    np.testing.assert_allclose(s["mean"], 3.0, atol=1e-5)  # mean of [1,3,5]
