import dataclasses
import datetime
from pathlib import Path

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.dataset.merged import MergeNoConcatDatasetConfig
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.xarray import XarrayDataConfig
from fme.downscaling.data.config import (
    DataLoaderConfig,
    PairedDataLoaderConfig,
    XarrayEnsembleDataConfig,
    _build_aligned_subset_pair,
    build_from_config_sequence,
)
from fme.downscaling.data.utils import ClosedInterval
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.test_utils import data_paths_helper
from fme.downscaling.typing_ import FineResCoarseResPair


def _write_global_nc(path: Path, n_lat: int, n_lon: int, num_timesteps: int) -> None:
    """Write a minimal NetCDF with global (0-360) lon coords and lat in 0-8 range.

    Each grid point's value encodes its source longitude (broadcast across time
    and lat), so a roll of the data can be checked against the rolled lon coords.
    """
    time_coord = [
        cftime.DatetimeProlepticGregorian(2000, 1, 1) + datetime.timedelta(days=i)
        for i in range(num_timesteps)
    ]
    lon_spacing = 360.0 / n_lon
    lat_spacing = 8.0 / n_lat
    lons = np.array(
        [lon_spacing / 2 + i * lon_spacing for i in range(n_lon)], dtype=np.float32
    )
    lats = np.array(
        [lat_spacing / 2 + i * lat_spacing for i in range(n_lat)], dtype=np.float32
    )

    data = (
        np.broadcast_to(lons[None, None, :], (num_timesteps, n_lat, n_lon))
        .astype(np.float32)
        .copy()
    )
    ds = xr.Dataset(
        {
            "var0": xr.DataArray(data, dims=["time", "lat", "lon"]),
            "var1": xr.DataArray(data, dims=["time", "lat", "lon"]),
        },
        coords={
            "time": xr.DataArray(time_coord, dims=["time"]),
            "lat": xr.DataArray(lats, dims=["lat"]),
            "lon": xr.DataArray(lons, dims=["lon"]),
        },
    )
    path.parent.mkdir(exist_ok=True)
    ds.to_netcdf(path, format="NETCDF4_CLASSIC")


def global_data_paths_helper(
    tmp_path: Path, num_timesteps: int = 4, scale_factor: int = 2
) -> FineResCoarseResPair[str]:
    """Like data_paths_helper but with global (0-360) lon coordinates.

    Coarse is 4 lat × 8 lon (45° lon spacing), fine is 8 lat × 16 lon
    (22.5° lon spacing) -- 2× scale factor in each spatial dimension.
    Lat midpoints are in [0, 8] so ClosedInterval(1, 4) works for lat_extent.
    """
    fine_path = tmp_path / "fine" / "data.nc"
    coarse_path = tmp_path / "coarse" / "data.nc"
    coarse_n_lon = 8
    coarse_n_lat = 4
    _write_global_nc(
        fine_path,
        n_lat=coarse_n_lat * scale_factor,
        n_lon=coarse_n_lon * scale_factor,
        num_timesteps=num_timesteps,
    )
    _write_global_nc(
        coarse_path, n_lat=coarse_n_lat, n_lon=coarse_n_lon, num_timesteps=num_timesteps
    )
    return FineResCoarseResPair[str](
        fine=str(tmp_path / "fine"), coarse=str(tmp_path / "coarse")
    )


@pytest.mark.parametrize(
    "fine_engine, coarse_engine, num_data_workers, expected",
    [
        ("netcdf4", "zarr", 0, None),
        ("netcdf4", "netcdf4", 2, None),
        ("netcdf4", "zarr", 2, "forkserver"),
        ("zarr", "zarr", 2, "forkserver"),
    ],
)
def test_DataLoaderConfig_mpcontext(
    fine_engine, coarse_engine, num_data_workers, expected
):
    fine_config = XarrayDataConfig(
        data_path="fine_dataset_path",
        engine=fine_engine,
        file_pattern="*.nc" if fine_engine == "netcdf4" else "a.zarr",
    )
    coarse_config = XarrayDataConfig(
        data_path="coarse_dataset_path",
        engine=coarse_engine,
        file_pattern="*.nc" if coarse_engine == "netcdf4" else "a.zarr",
    )
    loader_config = PairedDataLoaderConfig(
        fine=[fine_config],
        coarse=[coarse_config],
        batch_size=2,
        num_data_workers=num_data_workers,
        strict_ensemble=False,
    )
    assert loader_config._mp_context() == expected


@pytest.mark.medium_duration
def test_DataLoaderConfig_build(tmp_path):
    # TODO: this test can be removed after future PRs add a no-target
    # run script integration test that covers this functionality.
    paths = data_paths_helper(tmp_path)
    requirements = DataRequirements(
        fine_names=[], coarse_names=["var0"], n_timesteps=1, use_fine_topography=True
    )
    data_config = DataLoaderConfig(
        coarse=[XarrayDataConfig(paths.coarse)],
        batch_size=2,
        num_data_workers=1,
        strict_ensemble=False,
        lat_extent=ClosedInterval(1, 4),
        lon_extent=ClosedInterval(0, 3),
    )
    data = data_config.build(requirements=requirements)
    batch = next(iter(data.loader))
    # lat/lon midpoints are on (0.5, 1.5, ...)
    assert batch.data["var0"].shape == (2, 3, 3)


def test_XarrayEnsembleDataConfig():
    """Tests the XarrayEnsembleDataConfig class."""
    n_ensemble_members = 5
    base_config = XarrayDataConfig(
        data_path="ensemble_dataset_path", n_repeats=3, spatial_dimensions="healpix"
    )
    ensemble_config = XarrayEnsembleDataConfig(
        data_config=base_config,
        ensemble_dim="sample",
        n_ensemble_members=n_ensemble_members,
    )
    isel_sample_configs = [
        dataclasses.replace(base_config, isel={"sample": i})
        for i in range(n_ensemble_members)
    ]
    assert ensemble_config.expand() == isel_sample_configs


def test_PairedDataLoaderConfig_sample_with_replacement(tmp_path):
    paths = data_paths_helper(tmp_path)
    requirements = DataRequirements(
        fine_names=["var0"],
        coarse_names=["var0"],
        n_timesteps=1,
        use_fine_topography=False,
    )
    n_sample = 3
    data_config = PairedDataLoaderConfig(
        fine=[XarrayDataConfig(paths.fine)],
        coarse=[XarrayDataConfig(paths.coarse)],
        batch_size=1,
        num_data_workers=1,
        strict_ensemble=False,
        lat_extent=ClosedInterval(1, 4),
        lon_extent=ClosedInterval(0, 3),
        sample_with_replacement=n_sample,
    )
    data = data_config.build(requirements=requirements, train=True)
    assert len(data.loader) == n_sample


@pytest.mark.medium_duration
def test_DataLoaderConfig_includes_merge(tmp_path):
    """Test DataLoaderConfig with coarse as
    [XarrayDataConfig, MergeNoConcatDatasetConfig]."""
    paths = data_paths_helper(tmp_path, num_timesteps=4)
    requirements = DataRequirements(
        fine_names=[],
        coarse_names=["var0"],
        n_timesteps=1,
        use_fine_topography=True,
    )
    coarse_configs: list[XarrayDataConfig | MergeNoConcatDatasetConfig] = [
        XarrayDataConfig(paths.coarse),
        MergeNoConcatDatasetConfig(merge=[XarrayDataConfig(paths.coarse)]),
    ]
    data_config = DataLoaderConfig(
        coarse=coarse_configs,
        batch_size=2,
        num_data_workers=0,
        strict_ensemble=False,
        lat_extent=ClosedInterval(1, 4),
        lon_extent=ClosedInterval(0, 3),
    )

    data = data_config.build(requirements=requirements)
    # XarrayDataConfig + MergeNoConcatDatasetConfig each
    # contribute 4 timesteps = 8 total
    assert len(data.loader) == 4  # 8 samples / batch_size 2


def test_config_raise_error_on_invalid_lat_extent():
    with pytest.raises(ValueError):
        DataLoaderConfig(
            coarse=[XarrayDataConfig("coarse_dataset_path")],
            batch_size=1,
            num_data_workers=1,
            strict_ensemble=False,
            lat_extent=ClosedInterval(-90, 90),
            lon_extent=ClosedInterval(0, 3),
        )


def test_paired_config_raise_error_on_invalid_lat_extent():
    with pytest.raises(ValueError):
        PairedDataLoaderConfig(
            fine=[XarrayDataConfig("fine_dataset_path")],
            coarse=[XarrayDataConfig("fine_dataset_path")],
            batch_size=1,
            num_data_workers=1,
            strict_ensemble=False,
            lat_extent=ClosedInterval(-90, 90),
            lon_extent=ClosedInterval(0, 3),
        )


def test_config_raise_error_on_deprecated_topography():
    with pytest.raises(ValueError, match="deprecated"):
        DataLoaderConfig(
            coarse=[XarrayDataConfig("coarse_dataset_path")],
            batch_size=1,
            num_data_workers=1,
            strict_ensemble=False,
            topography="data.nc",
        )


def test_paired_config_raise_error_on_deprecated_topography():
    with pytest.raises(ValueError, match="deprecated"):
        PairedDataLoaderConfig(
            fine=[XarrayDataConfig("fine_dataset_path")],
            coarse=[XarrayDataConfig("coarse_dataset_path")],
            batch_size=1,
            num_data_workers=1,
            strict_ensemble=False,
            topography="data.nc",
        )


@pytest.mark.medium_duration
def test_PairedDataLoaderConfig_includes_merge(tmp_path):
    paths = data_paths_helper(tmp_path, num_timesteps=4)
    requirements = DataRequirements(
        fine_names=["var0"],
        coarse_names=["var0"],
        n_timesteps=1,
        use_fine_topography=False,
    )
    fine_configs: list[XarrayDataConfig | MergeNoConcatDatasetConfig] = [
        XarrayDataConfig(paths.fine),
        MergeNoConcatDatasetConfig(merge=[XarrayDataConfig(paths.fine)]),
    ]
    coarse_configs: list[XarrayDataConfig | MergeNoConcatDatasetConfig] = [
        XarrayDataConfig(paths.coarse),
        MergeNoConcatDatasetConfig(merge=[XarrayDataConfig(paths.coarse)]),
    ]
    data_config = PairedDataLoaderConfig(
        fine=fine_configs,
        coarse=coarse_configs,
        batch_size=2,
        num_data_workers=0,
        strict_ensemble=False,
        lat_extent=ClosedInterval(1, 4),
        lon_extent=ClosedInterval(0, 3),
    )
    data = data_config.build(requirements=requirements, train=True)
    # XarrayDataConfig + MergeNoConcatDatasetConfig each contribute
    # 4 timesteps = 8 total
    assert len(data.loader) == 4  # 8 samples / batch_size 2
    batch = next(iter(data.loader))
    assert batch.coarse.data["var0"].shape == (2, 3, 3)
    assert batch.fine.data["var0"].shape == (2, 6, 6)


@pytest.mark.parametrize(
    "scale_factor",
    [
        pytest.param(2, id="even-factor"),
        pytest.param(3, id="odd-factor-half-cell-anchor"),
    ],
)
def test_build_aligned_subset_pair_preserves_scale_factor_across_seam(
    tmp_path, scale_factor
):
    """Fine/coarse subsets must keep the scale factor for a seam-crossing extent.

    The odd scale factor is the non-trivial case: _roll_lons_to_extent_convention
    rolls the fine grid using a half-coarse-cell anchor, while
    HorizontalSubsetDataset re-derives its roll from the adjusted fine extent (an
    integer number of fine cells). For odd factors these anchors differ by half a
    fine cell, so this guards that the two independent rolls still agree.
    """
    paths = global_data_paths_helper(tmp_path, scale_factor=scale_factor)
    n_timesteps = IntSchedule.from_constant(1)
    dataset_fine, properties_fine = build_from_config_sequence(
        configs=[XarrayDataConfig(paths.fine)],
        names=["var0"],
        n_timesteps=n_timesteps,
        strict_ensemble=False,
    )
    dataset_coarse, properties_coarse = build_from_config_sequence(
        configs=[XarrayDataConfig(paths.coarse)],
        names=["var0"],
        n_timesteps=n_timesteps,
        strict_ensemble=False,
    )

    fine_subset, coarse_subset = _build_aligned_subset_pair(
        dataset_fine=dataset_fine,
        properties_fine=properties_fine,
        dataset_coarse=dataset_coarse,
        properties_coarse=properties_coarse,
        lat_extent=ClosedInterval(1.0, 4.0),
        lon_extent=ClosedInterval(-22.5, 22.5),
    )

    coarse_coords = coarse_subset.subset_latlon_coordinates
    fine_coords = fine_subset.subset_latlon_coordinates
    assert len(fine_coords.lat) == scale_factor * len(coarse_coords.lat)
    assert len(fine_coords.lon) == scale_factor * len(coarse_coords.lon)

    # Data values are the original longitudes, shifted only in position by the
    # roll, while the subset coords are in the requested interval's convention; mod
    # 360 compares them as the same physical longitude. This is the value-level
    # check at the odd factor -- where the config-side and dataset-side rolls use
    # different anchors -- which the scale-factor count alone could not catch.
    for subset, coords in (
        (fine_subset, fine_coords),
        (coarse_subset, coarse_coords),
    ):
        data_dict, *_ = subset[0]
        data = data_dict["var0"]  # (..., n_lat, n_lon); values == source lon
        expected = coords.lon.reshape(*([1] * (data.ndim - 1)), -1)
        assert torch.allclose(data % 360.0, expected % 360.0, atol=1e-3)


def test_PairedDataLoaderConfig_prime_meridian_crossing(tmp_path):
    """A seam-crossing extent must load with data and coordinates rolled together.

    Verifies the end-to-end loader: each value encodes its source longitude, so
    after the roll every lat row of the loaded data must still equal that grid's
    (rolled) lon coordinate. Scale-factor preservation across downscale factors is
    covered by test_build_aligned_subset_pair_preserves_scale_factor_across_seam.
    """
    paths = global_data_paths_helper(tmp_path)
    requirements = DataRequirements(
        fine_names=["var0"],
        coarse_names=["var0"],
        n_timesteps=1,
        use_fine_topography=False,
    )
    data_config = PairedDataLoaderConfig(
        fine=[XarrayDataConfig(paths.fine)],
        coarse=[XarrayDataConfig(paths.coarse)],
        batch_size=1,
        num_data_workers=0,
        strict_ensemble=False,
        lat_extent=ClosedInterval(1.0, 4.0),
        lon_extent=ClosedInterval(-22.5, 22.5),
    )

    data = data_config.build(requirements=requirements, train=False)
    batch = next(iter(data.loader))

    # The data values are the original longitudes, shifted only in position by the
    # roll (torch.roll never remaps values), while latlon_coordinates is expressed
    # in the requested interval's convention (e.g. original 337.5 -> -22.5).
    # Comparing mod 360 checks the two still refer to the same physical longitude
    # despite living in different 360-degree windows.
    for grid in (batch.coarse, batch.fine):
        lon = grid.latlon_coordinates.lon[0].cpu()  # batch members are identical
        values = grid.data["var0"][0].cpu()  # (n_lat, n_lon)
        expected = lon.unsqueeze(0).expand_as(values)
        assert torch.allclose(values % 360.0, expected % 360.0, atol=1e-3)
