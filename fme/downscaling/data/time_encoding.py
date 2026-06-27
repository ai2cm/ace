"""Per-frame physical-time features for video downscaling.

These are the inputs cBottle's ``CalendarEmbedding`` consumes (see
``cBottle/PHYSICAL_TIMESTEP_REPORT.md`` §3): each frame carries a
``day_of_year`` (seasonal cycle) and a ``second_of_day`` (diurnal cycle).

Following cBottle, the loader emits **UTC** values only; the longitude shift to
local solar time is performed inside the embedding module (model-side), so this
stays purely a data-loading concern.
"""

import numpy as np
import torch
import xarray as xr


def compute_calendar_features(
    time: xr.DataArray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-frame ``(day_of_year, second_of_day)`` for a clip.

    Args:
        time: 1-D ``DataArray`` of length ``T`` holding the wall-clock time of
            each frame (cftime or numpy-datetime values).

    Returns:
        ``(day_of_year, second_of_day)``, each a ``float32`` tensor of shape
        ``(T,)``. ``day_of_year`` is 1-based (1..365/366); ``second_of_day`` is
        in ``[0, 86400)``. Both are UTC.
    """
    try:
        # Works for both numpy-datetime and cftime-backed DataArrays.
        day_of_year = np.asarray(time.dt.dayofyear).reshape(-1)
        second_of_day = (
            np.asarray(time.dt.hour) * 3600
            + np.asarray(time.dt.minute) * 60
            + np.asarray(time.dt.second)
        ).reshape(-1)
    except (AttributeError, TypeError):
        # Fall back to raw cftime objects (which expose ``dayofyr``).
        values = np.asarray(time.values).reshape(-1)
        day_of_year = np.array([t.dayofyr for t in values])
        second_of_day = np.array(
            [t.hour * 3600 + t.minute * 60 + t.second for t in values]
        )

    return (
        torch.as_tensor(day_of_year, dtype=torch.float32),
        torch.as_tensor(second_of_day, dtype=torch.float32),
    )
