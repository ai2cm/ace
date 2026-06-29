"""Per-frame physical-time features (UTC ``day_of_year``/``second_of_day``)
for video downscaling, consumed by a cBottle-style ``CalendarEmbedding``."""

import numpy as np
import torch
import xarray as xr


def compute_calendar_features(
    time: xr.DataArray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-frame UTC ``(day_of_year, second_of_day)`` for a clip.

    Returns two ``float32`` tensors of shape ``(T,)``: 1-based day-of-year and
    second-of-day in ``[0, 86400)``.
    """
    try:
        day_of_year = np.asarray(time.dt.dayofyear).reshape(-1)
        second_of_day = (
            np.asarray(time.dt.hour) * 3600
            + np.asarray(time.dt.minute) * 60
            + np.asarray(time.dt.second)
        ).reshape(-1)
    except (AttributeError, TypeError):
        values = np.asarray(time.values).reshape(-1)
        day_of_year = np.array([t.dayofyr for t in values])
        second_of_day = np.array(
            [t.hour * 3600 + t.minute * 60 + t.second for t in values]
        )

    return (
        torch.as_tensor(day_of_year, dtype=torch.float32),
        torch.as_tensor(second_of_day, dtype=torch.float32),
    )
