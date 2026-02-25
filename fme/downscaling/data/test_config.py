import dataclasses

import pytest
import torch

from fme.core.dataset.merged import MergeNoConcatDatasetConfig
from fme.core.dataset.xarray import XarrayDataConfig
from fme.downscaling.data.config import (
    DataLoaderConfig,
    PairedDataLoaderConfig,
    XarrayEnsembleDataConfig,
)
from fme.downscaling.data.topography import StaticInput, StaticInputs
from fme.downscaling.data.utils import ClosedInterval, LatLonCoordinates
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.test_utils import data_paths_helper


def get_static_inputs(shape=(8, 8)):
    return StaticInputs(
        fields=[
            StaticInput(
                data=torch.ones(shape),
                coords=LatLonCoordinates(
                    lat=torch.ones(shape[0]), lon=torch.ones(shape[1])
                ),
            )
        ]
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


def test_DataLoaderConfig_build(tmp_path, very_fast_only: bool):
    # TODO: this test can be removed after future PRs add a no-target
    # run script integration test that covers this functionality.
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    paths = data_paths_helper(tmp_path)
    requirements = DataRequirements(
        fine_names=[], coarse_names=["var0"], n_timesteps=1, use_fine_topography=True
    )
    data_config = DataLoaderConfig(
        coarse=[XarrayDataConfig(paths.coarse)],
        batch_size=2,
        num_data_workers=1,
        strict_ensemble=False,
        topography=f"{paths.fine}/data.nc",
        lat_extent=ClosedInterval(1, 4),
        lon_extent=ClosedInterval(0, 3),
    )
    data = data_config.build(
        requirements=requirements, static_inputs=get_static_inputs(shape=(8, 8))
    )
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
        use_fine_topography=True,
    )
    n_sample = 3
    data_config = PairedDataLoaderConfig(
        fine=[XarrayDataConfig(paths.fine)],
        coarse=[XarrayDataConfig(paths.coarse)],
        batch_size=1,
        num_data_workers=1,
        strict_ensemble=False,
        topography=f"{paths.fine}/data.nc",
        lat_extent=ClosedInterval(1, 4),
        lon_extent=ClosedInterval(0, 3),
        sample_with_replacement=n_sample,
    )
    data = data_config.build(requirements=requirements, train=True)
    assert len(data.loader) == n_sample


def test_DataLoaderConfig_includes_merge(tmp_path, very_fast_only: bool):
    """Test DataLoaderConfig with coarse as
    [XarrayDataConfig, MergeNoConcatDatasetConfig]."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
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
        topography=f"{paths.fine}/data.nc",
        lat_extent=ClosedInterval(1, 4),
        lon_extent=ClosedInterval(0, 3),
    )

    data = data_config.build(
        requirements=requirements, static_inputs=get_static_inputs(shape=(8, 8))
    )
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
            topography=f"data.nc",
            lat_extent=ClosedInterval(-90, 90),
            lon_extent=ClosedInterval(0, 3),
        )


def test_PairedDataLoaderConfig_includes_merge(tmp_path, very_fast_only: bool):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    paths = data_paths_helper(tmp_path, num_timesteps=4)
    requirements = DataRequirements(
        fine_names=["var0"],
        coarse_names=["var0"],
        n_timesteps=1,
        use_fine_topography=True,
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
        topography=f"{paths.fine}/data.nc",
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
