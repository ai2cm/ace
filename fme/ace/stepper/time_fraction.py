"""Calendar position of a timestep, for time-of-year conditioning."""

import cftime
import numpy as np
import torch
import xarray as xr

from fme.ace.stepper.insolation.cm4 import CFTIME_TYPES, LENGTH_OF_YEAR
from fme.core.device import get_device

_YEAR_START = (1, 1, 1)


def compute_time_fraction(
    time: xr.DataArray, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Position within the calendar year, as a fraction in [0, 1).

    Calendar-aware: the year length is taken from the calendar of the data, so
    ``noleap`` and ``360_day`` datasets are handled correctly.

    Args:
        time: Times for a single timestep, of shape [n_batch].
        dtype: Floating point type of the result.

    Returns:
        A [n_batch] tensor on the current device.
    """
    values = np.asarray(time.values if isinstance(time, xr.DataArray) else time)
    first = values.ravel()[0]
    if not isinstance(first, cftime.datetime):
        raise TypeError(
            f"Expected cftime datetimes to derive the time of year, got {type(first)}."
        )
    calendar = first.calendar
    if calendar not in CFTIME_TYPES:
        raise ValueError(
            f"Unsupported calendar {calendar!r}, expected one of "
            f"{sorted(CFTIME_TYPES)}."
        )
    year_start = CFTIME_TYPES[calendar](*_YEAR_START)
    fraction = (values - year_start) / LENGTH_OF_YEAR[calendar]
    fraction = (fraction - np.floor(fraction)).astype(np.float64)
    return torch.as_tensor(fraction, device=get_device(), dtype=dtype)
