import numpy as np
import xarray as xr
from click.testing import CliRunner
from prepend_first_timestep_forcing import main


def _forcing():
    t = xr.date_range("1978-10-01", periods=4, freq="6h", use_cftime=False).values
    return xr.Dataset(
        {
            "surface_temperature": (
                ("time", "latitude", "longitude"),
                np.random.rand(4, 2, 2),
            ),
            "HGTsfc": (("latitude", "longitude"), np.ones((2, 2))),
            "ak_0": ((), np.float64(3.0)),
        },
        coords={"time": t, "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )


def test_statics_are_not_broadcast_along_time(tmp_path, monkeypatch):
    """Broadcasting statics stores one constant per timestep and slows the loader."""
    monkeypatch.setattr(xr, "open_zarr", lambda *a, **k: _forcing())
    out = tmp_path / "out.zarr"
    result = CliRunner().invoke(
        main,
        [
            str(out),
            "--input-forcing-path",
            "unused",
            "--input-timestamp",
            "1978-10-01T00:00:00",
            "--output-timestamp",
            "1978-09-30T18:00:00",
        ],
    )
    assert result.exit_code == 0, result.output

    ds = xr.open_dataset(out, engine="zarr")
    assert ds["HGTsfc"].dims == ("latitude", "longitude")
    assert ds["ak_0"].dims == ()
    assert ds["surface_temperature"].dims == ("time", "latitude", "longitude")
    assert ds.sizes["time"] == 5
    assert ds.time.values[0] == np.datetime64("1978-09-30T18:00:00")
