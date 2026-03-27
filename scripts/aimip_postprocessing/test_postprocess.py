import datetime

import cftime
import numpy as np
import pytest
import xarray as xr
from postprocess import (
    ace_layer_bounds,
    daily_data_time_coord,
    monthly_data_time_coord,
    nn_bounds,
    simplify_time_sample_dims,
    stack_vertical_dimension,
    time_bounds,
)

# --- stack_vertical_dimension ---


def test_stack_vertical_dimension_stacks_level_variables():
    ds = xr.Dataset(
        {
            "ta500": xr.DataArray(np.ones((3, 4)), dims=["lat", "lon"]),
            "ta850": xr.DataArray(np.ones((3, 4)) * 2, dims=["lat", "lon"]),
        }
    )
    result = stack_vertical_dimension(
        ds, "plev", "ta", r"[0-9]+$", "air_temperature", "air temperature", "K"
    )
    assert "ta" in result.data_vars
    assert "plev" in result.dims
    assert result["ta"].shape == (2, 3, 4)
    assert list(result["plev"].values) == [500.0, 850.0]


def test_stack_vertical_dimension_sorted_by_level():
    ds = xr.Dataset(
        {
            "ta850": xr.DataArray(np.full((2,), 2.0), dims=["x"]),
            "ta100": xr.DataArray(np.full((2,), 1.0), dims=["x"]),
            "ta500": xr.DataArray(np.full((2,), 0.5), dims=["x"]),
        }
    )
    result = stack_vertical_dimension(
        ds, "plev", "ta", r"[0-9]+$", "air_temperature", "air temperature", "K"
    )
    assert list(result["plev"].values) == [100.0, 500.0, 850.0]


def test_stack_vertical_dimension_single_variable_passthrough():
    data = xr.DataArray(np.ones((3, 4)), dims=["lat", "lon"])
    ds = xr.Dataset({"tas": data})
    result = stack_vertical_dimension(
        ds,
        "plev",
        "tas",
        r"[0-9]+$",
        "air_temperature",
        "air temperature at 2 meters",
        "K",
    )
    assert "tas" in result.data_vars
    assert "plev" not in result.dims


def test_stack_vertical_dimension_raises_for_multi_var_no_match():
    ds = xr.Dataset(
        {
            "tas": xr.DataArray(np.ones((3,)), dims=["x"]),
            "ps": xr.DataArray(np.ones((3,)), dims=["x"]),
        }
    )
    with pytest.raises(ValueError):
        stack_vertical_dimension(
            ds, "plev", "ta", r"[0-9]+$", "air_temperature", "air temperature", "K"
        )


# --- simplify_time_sample_dims ---


def _make_ds_with_sample_and_valid_time(n_time: int = 3) -> xr.Dataset:
    times = [cftime.datetime(2000, m, 15) for m in range(1, n_time + 1)]
    init_time = cftime.datetime(2000, 1, 1)
    da = xr.DataArray(
        np.ones((1, n_time, 4)),
        dims=["sample", "time", "x"],
        coords={
            "time": times,
            "valid_time": ("time", times),
            "init_time": init_time,
        },
    )
    return xr.Dataset({"var": da})


def test_simplify_time_sample_dims_removes_sample():
    ds = _make_ds_with_sample_and_valid_time()
    result = simplify_time_sample_dims(ds)
    assert "sample" not in result.dims


def test_simplify_time_sample_dims_promotes_valid_time():
    ds = _make_ds_with_sample_and_valid_time()
    result = simplify_time_sample_dims(ds)
    assert "time" in result.coords
    assert "valid_time" not in result.coords


def test_simplify_time_sample_dims_drops_init_time():
    ds = _make_ds_with_sample_and_valid_time()
    result = simplify_time_sample_dims(ds)
    assert "init_time" not in result.coords


# --- monthly_data_time_coord ---


def _make_monthly_ds(counts: list[int]) -> xr.Dataset:
    n = len(counts)
    times = [
        cftime.datetime(2000, 1, 15) + datetime.timedelta(days=30 * i) for i in range(n)
    ]
    ds = xr.Dataset(
        {
            "var": xr.DataArray(np.ones(n), dims=["time"]),
            "counts": xr.DataArray(np.array(counts, dtype=float), dims=["time"]),
        },
        coords={"time": times},
    )
    return ds


def test_monthly_data_time_coord_drops_zero_count_months():
    ds = _make_monthly_ds([1, 0, 1])
    result = monthly_data_time_coord(ds)
    assert len(result.time) == 2


def test_monthly_data_time_coord_drops_counts_variable():
    ds = _make_monthly_ds([1, 1])
    result = monthly_data_time_coord(ds)
    assert "counts" not in result


def test_monthly_data_time_coord_shifts_to_first_of_month():
    ds = _make_monthly_ds([1])
    result = monthly_data_time_coord(ds)
    t = result.time.values[0]
    assert t.day == 1


# --- daily_data_time_coord ---


def _make_daily_ds(n: int = 3) -> xr.Dataset:
    times = [
        cftime.datetime(2000, 1, 1, 9) + datetime.timedelta(days=i) for i in range(n)
    ]
    return xr.Dataset(
        {"var": xr.DataArray(np.ones(n), dims=["time"])},
        coords={"time": times},
    )


def test_daily_data_time_coord_shifts_to_midnight():
    ds = _make_daily_ds()
    result = daily_data_time_coord(ds)
    for t in result.time.values:
        assert t.hour == 0


def test_daily_data_time_coord_preserves_count():
    ds = _make_daily_ds(5)
    result = daily_data_time_coord(ds)
    assert len(result.time) == 5


# --- nn_bounds ---


def test_nn_bounds_shape():
    centers = np.array([0.0, 1.0, 2.0, 3.0])
    bounds = nn_bounds(centers, -0.5, 3.5)
    assert bounds.shape == (4, 2)


def test_nn_bounds_start_end():
    centers = np.array([0.0, 1.0, 2.0])
    bounds = nn_bounds(centers, -0.5, 2.5)
    assert bounds[0, 0] == -0.5
    assert bounds[-1, 1] == 2.5


def test_nn_bounds_interior_midpoints():
    centers = np.array([0.0, 2.0, 6.0])
    bounds = nn_bounds(centers, -1.0, 8.0)
    assert bounds[0, 1] == pytest.approx(1.0)
    assert bounds[1, 0] == pytest.approx(1.0)
    assert bounds[1, 1] == pytest.approx(4.0)
    assert bounds[2, 0] == pytest.approx(4.0)


# --- time_bounds ---


def test_time_bounds_shape():
    starts = np.array([cftime.datetime(2000, m, 1) for m in range(1, 5)])
    bounds = time_bounds(starts, (2000, 5, 1))
    assert bounds.shape == (4, 2)


def test_time_bounds_last_entry_uses_end_time():
    starts = np.array([cftime.datetime(2000, 1, 1), cftime.datetime(2000, 2, 1)])
    end = (2000, 3, 1)
    bounds = time_bounds(starts, end)
    assert bounds[-1, 1] == cftime.datetime(*end)


def test_time_bounds_consecutive_periods():
    starts = np.array([cftime.datetime(2000, m, 1) for m in range(1, 4)])
    bounds = time_bounds(starts, (2000, 4, 1))
    assert bounds[0, 1] == bounds[1, 0]
    assert bounds[1, 1] == bounds[2, 0]


# --- ace_layer_bounds ---


def test_ace_layer_bounds_has_eight_layers():
    result = ace_layer_bounds("model_layer", "bnds")
    assert result.shape[0] == 8


def test_ace_layer_bounds_starts_at_zero():
    result = ace_layer_bounds("model_layer", "bnds")
    assert result.values[0, 0] == pytest.approx(0.0)


def test_ace_layer_bounds_ends_at_surface():
    result = ace_layer_bounds("model_layer", "bnds")
    assert result.values[-1, 1] == pytest.approx(100000.0)


def test_ace_layer_bounds_monotonically_increasing():
    result = ace_layer_bounds("model_layer", "bnds")
    flat = result.values.flatten()
    # Adjacent layers share interface pressures, so equal values are expected at boundaries.  # noqa: E501
    assert all(flat[i] <= flat[i + 1] for i in range(len(flat) - 1))
