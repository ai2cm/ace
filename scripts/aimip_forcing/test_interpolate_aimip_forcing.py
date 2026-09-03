import interpolate_aimip_forcing as mod
import numpy as np
import xarray as xr
from click.testing import CliRunner
from interpolate_aimip_forcing import (
    DEFAULT_EXTENSION_START,
    SURFACE_TEMPERATURE_NAME,
    main,
)


def _times(start, end, freq):
    return xr.DataArray(
        xr.date_range(start=start, end=end, freq=freq, use_cftime=False).values,
        dims=["time"],
        name="time",
    )


def _fake_era5(start, end, freq="6h"):
    t = xr.date_range(start=start, end=end, freq=freq, use_cftime=False).values
    shape = (t.size, 2, 2)
    return xr.Dataset(
        {
            "DSWRFtoa": (("time", "latitude", "longitude"), np.zeros(shape)),
            "HGTsfc": (("time", "latitude", "longitude"), np.zeros(shape)),
            "land_fraction": (("time", "latitude", "longitude"), np.zeros(shape)),
        },
        coords={"time": t, "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )


def test_disabled_extension_errors_when_source_falls_short(tmp_path, monkeypatch):
    """Without an extension a short source would silently truncate the output."""
    monthly = xr.Dataset(
        {
            SURFACE_TEMPERATURE_NAME: (
                ("time", "latitude", "longitude"),
                np.zeros((2, 2, 2)),
            )
        },
        coords={
            "time": xr.date_range(
                "1979-01-01", periods=2, freq="MS", use_cftime=False
            ).values,
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        },
    )
    monkeypatch.setattr(mod, "open_aimip_forcing_data", lambda *a, **k: monthly)
    monkeypatch.setattr(
        mod,
        "get_existing_era5_forcing",
        lambda *a, **k: _fake_era5("1979-01-01", "1979-01-02"),
    )
    src = tmp_path / "in.nc"
    src.write_bytes(b"")

    result = CliRunner().invoke(
        main,
        [
            str(src),
            str(tmp_path / "out.zarr"),
            "--start-time",
            "1979-01-01T00:00:00",
            "--end-time",
            "1979-06-01T00:00:00",
            "--extension-start",
            "",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "before --end-time" in str(result.exception)


def test_extension_is_off_by_default():
    """Extension is opt-in; sources are expected to span the full window."""
    assert DEFAULT_EXTENSION_START == ""
