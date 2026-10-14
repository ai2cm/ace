import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner
from create_aimip_ic_datasets import (
    NEAR_SURFACE_VARIABLES,
    PROGNOSTIC_VARIABLES,
    create_ic,
    main,
)

TARGET = "1978-09-30T18:00:00"


def _era5(times, names):
    n_lat, n_lon = 4, 8
    return xr.Dataset(
        {
            name: (
                ("time", "latitude", "longitude"),
                np.random.rand(len(times), n_lat, n_lon),
            )
            for name in names
        },
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "latitude": np.arange(n_lat, dtype=float),
            "longitude": np.arange(n_lon, dtype=float),
        },
    )


def test_create_ic_keeps_time_as_length_one_dimension():
    """get_initial_condition requires shape (n_samples, [spatial dims]), so `time`
    must survive selection as a dimension rather than collapsing to a scalar."""
    era5 = _era5(["1978-09-29T00", "1978-09-30T00"], PROGNOSTIC_VARIABLES)
    ic = create_ic(era5, "1978-09-29T00", np.datetime64(TARGET))
    assert ic.sizes["time"] == 1
    for name in PROGNOSTIC_VARIABLES:
        assert ic[name].dims == ("time", "latitude", "longitude")


def test_create_ic_restamps_time_to_target():
    era5 = _era5(["1978-09-29T00", "1978-09-30T00"], PROGNOSTIC_VARIABLES)
    ic = create_ic(era5, "1978-09-30T00", np.datetime64(TARGET))
    assert ic.time.values[0] == np.datetime64(TARGET)


def test_create_ic_selects_the_requested_timestamp():
    era5 = _era5(["1978-09-29T00", "1978-09-30T00"], PROGNOSTIC_VARIABLES)
    ic = create_ic(era5, "1978-09-30T00", np.datetime64(TARGET))
    expected = era5[PROGNOSTIC_VARIABLES[0]].sel(time="1978-09-30T00").values
    np.testing.assert_array_equal(ic[PROGNOSTIC_VARIABLES[0]].values[0], expected)


@pytest.mark.parametrize(
    "flag, expect_near_surface",
    [([], False), (["--include-near-surface"], True)],
)
def test_near_surface_variables_included_only_on_request(
    tmp_path, monkeypatch, flag, expect_near_surface
):
    era5 = _era5(["1978-09-29T00"], PROGNOSTIC_VARIABLES + NEAR_SURFACE_VARIABLES)
    monkeypatch.setattr(xr, "open_zarr", lambda *a, **k: era5)

    result = CliRunner().invoke(
        main,
        [str(tmp_path), "--ic-timestamp", "1978-09-29T00", *flag],
    )
    assert result.exit_code == 0, result.output

    written = xr.load_dataset(tmp_path / "1978-09-30_IC0.nc")
    for name in PROGNOSTIC_VARIABLES:
        assert name in written
    for name in NEAR_SURFACE_VARIABLES:
        assert (name in written) is expect_near_surface
