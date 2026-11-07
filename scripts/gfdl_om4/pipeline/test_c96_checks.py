"""``stores`` checks C1–C3 on toy C96 and direct stores.

J: toy Gaussian cell, 1 × 4 cells, one timestep pair. Both stores hold the
same ocean area 6 m² and the same field totals; the C96 store spreads the
ocean of J = 1 into J = 2 (a ``gained_band`` cell), where the direct store
is land with NaN fields.
"""

import numpy as np
import pytest
import xarray as xr

for _name in ("xesmf", "esmpy", "apache_beam", "xarray_beam", "obstore"):
    pytest.importorskip(_name)

from .c96_checks import stores_checks
from .native_area import AREA_RTOL

NATIVE_WET_AREA = 6.0


def _store(regrid_path, areacello, ssf, tos, ice) -> xr.Dataset:
    dims = ("time", "lat", "lon")

    def field(x):
        return xr.DataArray(np.array(x, dtype=np.float32).reshape(1, 1, -1), dims=dims)

    return xr.Dataset(
        {
            "areacello": (("lat", "lon"), np.array([areacello], dtype=np.float32)),
            "sea_surface_fraction": (("lat", "lon"), np.array([ssf], dtype=np.float32)),
            "tos": field(tos),
            "sea_ice_fraction": field(ice),
        },
        coords={"time": [0]},
        attrs={"regrid_path": regrid_path},
    )


def _direct():
    # ocean area 2 + 4 + 0 + 0; Σ A s tos = 2·1 + 4·−2 = −6; Σ A ice = 2·0.5
    return _store(
        "direct",
        [2.0, 4.0, 3.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, -2.0, np.nan, np.nan],
        [0.5, 0.0, np.nan, np.nan],
    )


def _c96(tos_scale=1.0):
    # ocean area 2 + 3 + 1 + 0; Σ A s tos = 2·1 + 3·−3 + 1·1 = −6; Σ A ice = 1
    return _store(
        "c96",
        [2.0, 3.0, 2.0, 1.0],
        [1.0, 1.0, 0.5, 0.0],
        [1.0, -3.0 * tos_scale, 1.0, np.nan],
        [0.25, 0.0, 0.25, np.nan],
    )


def _checks(c96, direct=None, native=NATIVE_WET_AREA):
    return stores_checks(
        c96,
        _direct() if direct is None else direct,
        native,
        ["tos"],
        ["sea_ice_fraction"],
    )


def test_equal_totals_pass():
    result = _checks(_c96())
    assert result["pass"]
    assert result["C1"]["c96path"]["pass"] and result["C1"]["direct"]["pass"]
    assert result["C2"]["tos"]["lhs"] == pytest.approx([-6.0])
    # scale: max_p Σ_J A s |tos| = max(2 + 8, 2 + 9 + 1)
    assert result["C2"]["tos"]["scale"] == pytest.approx([12.0])
    assert result["C3"]["sea_ice_fraction"]["rhs"] == pytest.approx([1.0])


def test_gained_band_share():
    result = _checks(_c96())
    assert result["reports"]["gained_band_n_cells"] == 1
    assert result["reports"]["gained_band_ocean_area_share"] == pytest.approx(1 / 6)


def test_field_total_mismatch_fails():
    result = _checks(_c96(tos_scale=1 + 100 * AREA_RTOL))
    assert not result["C2"]["tos"]["pass"]
    assert result["C1"]["c96path"]["pass"]
    assert not result["pass"]


def test_native_total_mismatch_fails():
    result = _checks(_c96(), native=NATIVE_WET_AREA * (1 + 100 * AREA_RTOL))
    assert not result["C1"]["c96path"]["pass"]
    assert not result["pass"]


def test_swapped_stores_raise():
    with pytest.raises(ValueError, match="regrid_path"):
        _checks(_direct(), direct=_c96())


def test_misaligned_times_raise():
    with pytest.raises(ValueError):
        _checks(_c96().assign_coords(time=[1]))
