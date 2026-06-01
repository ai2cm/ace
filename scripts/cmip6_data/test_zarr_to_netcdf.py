"""Smoke tests for ``zarr_to_netcdf.convert_one``.

The downstream training-side loader globs ``data.*.nc`` and sorts
lexicographically, so the only contract this script has to honor is:

1. Each input timestep lands in exactly one output file.
2. File names are ``data.{first_year}-{last_year}.nc`` and sort in
   chronological order under plain string sort.
3. Intra-file time-axis chunking stays at 1 day regardless of how
   many years fit in one file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))


def _make_zarr(zarr_path: Path, year_range: tuple[int, int]) -> int:
    """Write a tiny multi-year zarr to ``zarr_path`` and return the
    number of timesteps written."""
    start, end = year_range
    times = xr.date_range(
        f"{start}-01-01",
        f"{end}-12-31",
        freq="D",
        calendar="noleap",
    )
    nt = len(times)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "TMP2m": xr.DataArray(
                rng.normal(280, 5, size=(nt, 4, 8)).astype(np.float32),
                dims=("time", "lat", "lon"),
                coords={"time": times},
            ),
        }
    )
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True, zarr_format=3)
    return nt


def test_convert_one_decade_files(tmp_path: Path):
    """Default ``years_per_file=10`` produces decade-bounded files
    named ``data.YYYY-YYYY.nc`` whose union covers every input
    timestep exactly once."""
    from zarr_to_netcdf import convert_one

    zarr_path = tmp_path / "src.zarr"
    n_input = _make_zarr(zarr_path, (1940, 1959))  # 20 years
    nc_dir = tmp_path / "out"

    convert_one(str(zarr_path), str(nc_dir), years_per_file=10)

    nc_files = sorted(p.name for p in nc_dir.glob("*.nc"))
    # 20 years, decade-bounded: 1940-1949 + 1950-1959.
    assert nc_files == ["data.1940-1949.nc", "data.1950-1959.nc"], nc_files

    # Union must round-trip with same timestep count.
    combined = xr.open_mfdataset(sorted(nc_dir.glob("*.nc")), combine="by_coords")
    assert combined.sizes["time"] == n_input
    combined.close()


def test_convert_one_custom_years_per_file(tmp_path: Path):
    """``years_per_file=5`` produces 5-year files; ``=1`` produces
    legacy yearly files with the (now self-consistent)
    ``data.{year}-{year}.nc`` naming."""
    from zarr_to_netcdf import convert_one

    zarr_path = tmp_path / "src.zarr"
    _make_zarr(zarr_path, (2000, 2009))  # 10 years
    nc_dir_5 = tmp_path / "out_5"
    nc_dir_1 = tmp_path / "out_1"

    convert_one(str(zarr_path), str(nc_dir_5), years_per_file=5)
    convert_one(str(zarr_path), str(nc_dir_1), years_per_file=1)

    files_5 = sorted(p.name for p in nc_dir_5.glob("*.nc"))
    assert files_5 == ["data.2000-2004.nc", "data.2005-2009.nc"]

    files_1 = sorted(p.name for p in nc_dir_1.glob("*.nc"))
    assert files_1 == [f"data.{y}-{y}.nc" for y in range(2000, 2010)]


def test_convert_one_partial_decade_at_boundary(tmp_path: Path):
    """A year span that straddles a decade boundary (e.g. 2015-2100,
    matching the SSP scenarios) still groups by floor-decade and
    names files by the actual first/last year present, not the
    nominal decade boundary."""
    from zarr_to_netcdf import convert_one

    zarr_path = tmp_path / "src.zarr"
    _make_zarr(zarr_path, (2015, 2024))  # straddles 2010s + 2020s
    nc_dir = tmp_path / "out"

    convert_one(str(zarr_path), str(nc_dir), years_per_file=10)

    files = sorted(p.name for p in nc_dir.glob("*.nc"))
    # 2015-2019 falls in the 2010s decade bin; 2020-2024 in the 2020s.
    # Filenames reflect the actual first/last year covered.
    assert files == ["data.2015-2019.nc", "data.2020-2024.nc"]


def test_convert_one_per_day_chunking_preserved(tmp_path: Path):
    """Inner HDF5 chunking must stay at 1 timestep per chunk
    regardless of ``years_per_file`` — the whole point of the
    netCDF mirror is fast random-access per day, and that depends
    on the chunk layout, not the file layout."""
    from zarr_to_netcdf import convert_one

    zarr_path = tmp_path / "src.zarr"
    _make_zarr(zarr_path, (1990, 2009))  # 20 years
    nc_dir = tmp_path / "out"

    convert_one(str(zarr_path), str(nc_dir), years_per_file=10)

    # xarray's netCDF backend surfaces the stored HDF5 chunk tuple in
    # ``encoding["chunksizes"]``. Reading h5py directly would be more
    # authoritative but the dev env's h5py is binary-mismatched
    # against the libhdf5 netcdf4 ships with (h5py 2.0 headers vs
    # libhdf5 1.14 runtime), so we use xarray which handles the
    # version drift cleanly.
    ds = xr.open_dataset(nc_dir / "data.1990-1999.nc", engine="netcdf4")
    chunks = ds["TMP2m"].encoding["chunksizes"]
    # Time dim is chunk size 1; spatial dims are full extent.
    assert chunks[0] == 1, chunks
    assert tuple(chunks[1:]) == ds["TMP2m"].shape[1:], chunks
    ds.close()


def test_convert_one_skips_existing_without_force(tmp_path: Path):
    """Without ``force``, an existing output netCDF is left alone.
    Pre-create a sentinel file with one-byte content; ``convert_one``
    must leave it untouched (it would otherwise be replaced by a
    real conversion of the input zarr).
    """
    from zarr_to_netcdf import convert_one

    zarr_path = tmp_path / "src.zarr"
    _make_zarr(zarr_path, (1940, 1949))
    nc_dir = tmp_path / "out"
    nc_dir.mkdir()
    sentinel = nc_dir / "data.1940-1949.nc"
    sentinel.write_bytes(b"x")

    convert_one(str(zarr_path), str(nc_dir), years_per_file=10)

    assert sentinel.read_bytes() == b"x"


def test_convert_one_overwrites_with_force(tmp_path: Path):
    """With ``force=True``, ``convert_one`` deletes the stale output
    and writes a fresh conversion in its place. The resulting file
    must be a real netCDF that round-trips the input timesteps —
    not the 1-byte sentinel we seeded.
    """
    from zarr_to_netcdf import convert_one

    zarr_path = tmp_path / "src.zarr"
    n_input = _make_zarr(zarr_path, (1940, 1949))
    nc_dir = tmp_path / "out"
    nc_dir.mkdir()
    sentinel = nc_dir / "data.1940-1949.nc"
    sentinel.write_bytes(b"x")

    convert_one(str(zarr_path), str(nc_dir), years_per_file=10, force=True)

    # Real netCDF now, not the sentinel.
    ds = xr.open_dataset(sentinel)
    assert ds.sizes["time"] == n_input
    ds.close()
