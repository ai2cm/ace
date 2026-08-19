import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.stepper.time_fraction import compute_time_fraction


def _times(dates, calendar="noleap"):
    ctor = {
        "noleap": cftime.DatetimeNoLeap,
        "360_day": cftime.Datetime360Day,
        "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    }[calendar]
    return xr.DataArray(np.array([ctor(*date) for date in dates]), dims=["sample"])


def test_january_first_is_zero():
    result = compute_time_fraction(_times([(2000, 1, 1)]))
    torch.testing.assert_close(result, torch.zeros(1), atol=1e-6, rtol=0)


def test_fraction_is_in_unit_interval_across_the_year():
    dates = [(2000, month, 15) for month in range(1, 13)]
    result = compute_time_fraction(_times(dates))
    assert torch.all(result >= 0.0)
    assert torch.all(result < 1.0)
    # Monotonic within a single year.
    assert torch.all(result[1:] > result[:-1])


def test_midyear_is_about_one_half():
    """Day 183 of a 365-day calendar is roughly halfway through the year."""
    result = compute_time_fraction(_times([(2000, 7, 2)]))
    assert 0.49 < float(result[0]) < 0.51


def test_same_day_in_different_years_matches():
    """The value must depend only on position within the year."""
    result = compute_time_fraction(_times([(1985, 3, 21), (2011, 3, 21)]))
    torch.testing.assert_close(result[0], result[1], atol=1e-3, rtol=0)


def test_360_day_calendar_uses_its_own_year_length():
    """Day 180 of a 360-day year is exactly halfway."""
    result = compute_time_fraction(_times([(2000, 7, 1)], calendar="360_day"))
    torch.testing.assert_close(result, torch.tensor([0.5]), atol=1e-6, rtol=0)


def test_calendars_disagree_on_the_same_date():
    """A 360-day year and a 365-day year place the same date differently."""
    noleap = compute_time_fraction(_times([(2000, 7, 1)]))
    day360 = compute_time_fraction(_times([(2000, 7, 1)], calendar="360_day"))
    assert abs(float(noleap[0]) - float(day360[0])) > 1e-3


def test_non_cftime_input_raises():
    """Integer times cannot carry calendar information, so fail loudly."""
    times = xr.DataArray(np.arange(4), dims=["sample"])
    with pytest.raises(TypeError, match="cftime"):
        compute_time_fraction(times)


def test_batch_shape_is_preserved():
    dates = [(2000, 1, 1), (2000, 4, 1), (2000, 8, 1), (2000, 12, 1)]
    assert compute_time_fraction(_times(dates)).shape == (4,)
