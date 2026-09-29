import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.dataset import derived
from fme.core.dataset.merged import (
    MergedXarrayDataset,
    MergeNoConcatDatasetConfig,
    get_merged_datasets,
    get_per_dataset_names,
)
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.xarray import XarrayDataConfig, XarrayDataset
from fme.core.frozen_mass_budget import (
    calving_residue,
    frozen_mass,
    target_frozen_mass_energy_budget_residual,
)

ATMOS = [
    "DSWRFsfc",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
    "total_frozen_precipitation_rate",
]
OCEAN = [
    "sst",
    "ocean_sea_ice_fraction",
    "simass",
    "sisnmass",
    "hflso",
    "evs",
    "prsn",
    "hfrunoffds",
    "hfds_total_area",
]
STATIC = ["land_fraction", "sea_surface_fraction"]
N_TIME, N_LAT, N_LON = 12, 4, 8
DERIVED = ["frozen_mass", "calving_residue", "frozen_mass_energy_budget_residual"]
_OCEAN = torch.ones(N_LAT, N_LON, dtype=torch.bool)
_OCEAN[0, 0] = False  # the all-land cell, NaN in every ocean field
# mask_sea_ice_fraction: 0 on land and on one ocean cell with mask_2d == 1
_ICE = _OCEAN.clone()
_ICE[1, 2] = False
_ICE_MASKED = ["frozen_mass", "frozen_mass_energy_budget_residual"]


def _write_stores(
    tmp_path, ice_mask: bool = True
) -> tuple[MergeNoConcatDatasetConfig, xr.Dataset]:
    """An atmosphere store (C) and an ocean store (O) on a 5-daily axis; ocean
    fields are NaN on the one all-land cell. O holds ``mask_2d`` (``_OCEAN``)
    and, if ``ice_mask``, ``mask_sea_ice_fraction`` (``_ICE``). Returns the
    config and the joint data.
    """
    rng = np.random.default_rng(0)
    time = xr.date_range(
        "2000-01-01", periods=N_TIME, freq="5D", calendar="noleap", use_cftime=True
    )
    coords = {
        "time": time,
        "lat": np.linspace(-60, 60, N_LAT).astype(np.float32),
        "lon": np.arange(N_LON, dtype=np.float32),
    }
    land = np.zeros((N_LAT, N_LON), dtype=np.float32)
    land[0, 0] = 1.0
    land[2, 3] = 0.5
    dims = ("time", "lat", "lon")

    def field(scale, offset=0.0, nan_land=False):
        x = (offset + scale * rng.random((N_TIME, N_LAT, N_LON))).astype(np.float32)
        if nan_land:
            x[:, 0, 0] = np.nan
        return xr.DataArray(x, dims=dims)

    static = {
        "land_fraction": xr.DataArray(land, dims=("lat", "lon")),
        "sea_surface_fraction": xr.DataArray(1 - land, dims=("lat", "lon")),
    }
    atmos = {name: field(100.0) for name in ATMOS}
    atmos["PRATEsfc"] = field(1e-4)
    atmos["total_frozen_precipitation_rate"] = field(1e-5)
    ocean = {
        "sst": field(10.0, 270.0, nan_land=True),
        "ocean_sea_ice_fraction": field(1.0, nan_land=True),
        "simass": field(900.0, nan_land=True),
        "sisnmass": field(300.0, nan_land=True),
        "hflso": field(20.0, -10.0, nan_land=True),
        "evs": field(1e-5, nan_land=True),
        "prsn": field(1e-5, nan_land=True),
        "hfrunoffds": field(5.0, nan_land=True),
        "hfds_total_area": field(100.0, -50.0, nan_land=True),
    }
    masks = {"mask_2d": xr.DataArray(_OCEAN.numpy().astype(np.float32), dims=dims[1:])}
    if ice_mask:
        masks["mask_sea_ice_fraction"] = xr.DataArray(
            _ICE.numpy().astype(np.float32), dims=dims[1:]
        )
    c_dir, o_dir = tmp_path / "c", tmp_path / "o"
    c_dir.mkdir()
    o_dir.mkdir()
    xr.Dataset({**atmos, **static}, coords=coords).to_netcdf(c_dir / "data.nc")
    xr.Dataset({**ocean, **static, **masks}, coords=coords).to_netcdf(o_dir / "data.nc")
    config = MergeNoConcatDatasetConfig(
        merge=[
            XarrayDataConfig(data_path=str(c_dir)),
            XarrayDataConfig(data_path=str(o_dir)),
        ]
    )
    joint = xr.Dataset({**atmos, **ocean, **static, **masks}, coords=coords)
    return config, joint


def _expected(joint: xr.Dataset, start: int, n: int) -> dict[str, torch.Tensor]:
    window = {
        name: torch.as_tensor(
            np.broadcast_to(joint[name].values, (N_TIME, N_LAT, N_LON))[
                start : start + n
            ].copy()
        )
        for name in ATMOS + OCEAN + STATIC
    }
    dt = 5 * 86400.0
    land = torch.isnan(window["sst"])
    expected = {
        "frozen_mass": frozen_mass(
            window["simass"], window["sisnmass"], window["sea_surface_fraction"]
        ),
        "calving_residue": calving_residue(
            window["hflso"], window["evs"], window["prsn"]
        ),
        "frozen_mass_energy_budget_residual": target_frozen_mass_energy_budget_residual(
            window, dt
        ),
    }
    out = {k: torch.where(land, float("nan"), v) for k, v in expected.items()}
    if "mask_sea_ice_fraction" in joint:
        for k in _ICE_MASKED:
            out[k] = torch.where(~_ICE, float("nan"), out[k])
    return out


def _build(config, names, n_timesteps=4):
    return get_merged_datasets(
        config, names, IntSchedule(start_value=n_timesteps, milestones=[])
    )


def test_expand_names():
    stored, names = derived.expand_names(["sst", "frozen_mass", "calving_residue"])
    assert names == ["frozen_mass", "calving_residue"]
    assert stored[0] == "sst"
    assert set(stored) == {"sst", "simass", "sisnmass", "sea_surface_fraction"} | {
        "hflso",
        "evs",
        "prsn",
    }
    assert len(stored) == len(set(stored))


def test_derived_names_match_recomputation(tmp_path):
    config, joint = _write_stores(tmp_path)
    requested = ["sst", "hfrunoffds", *DERIVED]
    dataset, properties = _build(config, requested)
    for idx in [0, 3]:
        tensors, time, *_ = dataset[idx]
        assert set(tensors) == set(requested)
        expected = _expected(joint, start=idx, n=4)
        for name in DERIVED:
            torch.testing.assert_close(tensors[name], expected[name], equal_nan=True)
        residual = tensors["frozen_mass_energy_budget_residual"]
        assert (residual[0, _ICE] == 0).all()
        assert (residual[1:, _ICE] != 0).any()
    for name in DERIVED:
        assert properties.variable_metadata[name] == derived.DERIVED_METADATA[name]
        assert (
            dataset.properties.variable_metadata[name]
            == (derived.DERIVED_METADATA[name])
        )


def test_derived_names_nan_where_stored_inputs_nan(tmp_path):
    """03a-c01: the loss masks where the target is NaN, so a derived target is
    NaN on land like the stored ocean fields it is built from."""
    config, _ = _write_stores(tmp_path)
    dataset, _ = _build(config, DERIVED)
    tensors, *_ = dataset[1]
    for name in DERIVED:
        valid = _ICE if name in _ICE_MASKED else _OCEAN
        assert torch.isnan(tensors[name][:, ~valid]).all(), name
        assert torch.isfinite(tensors[name][:, valid]).all(), name


def test_masked_names_carry_sea_ice_fraction_mask(tmp_path):
    """07: frozen_mass and the residual get mask_<name> == mask_sea_ice_fraction
    in the merged properties, and are NaN at every time where it is 0; values
    where it is 1 are those without the mask (03a-c01)."""
    config, joint = _write_stores(tmp_path)
    dataset, properties = _build(config, ["sst", *DERIVED])
    ice = torch.as_tensor(joint["mask_sea_ice_fraction"].values)
    mask_2d = torch.as_tensor(joint["mask_2d"].values)
    for props in (properties, dataset.properties):
        provider = props.spatial_mask_provider
        for name in _ICE_MASKED:
            assert f"mask_{name}" in provider.masks
            torch.testing.assert_close(provider.get_mask_tensor_for(name), ice)
        torch.testing.assert_close(
            provider.get_mask_tensor_for("calving_residue"), mask_2d
        )
        assert "mask_calving_residue" not in provider.masks
    tensors, *_ = dataset[2]
    reference = _expected(joint.drop_vars("mask_sea_ice_fraction"), start=2, n=4)
    for name in _ICE_MASKED:
        assert torch.isnan(tensors[name][:, ice == 0]).all(), name
        torch.testing.assert_close(
            tensors[name][:, ice == 1], reference[name][:, ice == 1]
        )
    torch.testing.assert_close(
        tensors["calving_residue"], reference["calving_residue"], equal_nan=True
    )


def test_absent_sea_ice_fraction_mask_raises(tmp_path):
    config, _ = _write_stores(tmp_path, ice_mask=False)
    for name in _ICE_MASKED:
        with pytest.raises(ValueError, match="mask_sea_ice_fraction"):
            _build(config, [name])
    dataset, properties = _build(config, ["calving_residue"])
    assert set(properties.spatial_mask_provider.masks) == {"mask_2d"}


@pytest.mark.parametrize(
    "name, nan_input",
    [
        ("frozen_mass", "sisnmass"),
        ("calving_residue", "prsn"),
        ("frozen_mass_energy_budget_residual", "hfds_total_area"),
        ("frozen_mass_energy_budget_residual", "simass"),
    ],
)
def test_apply_nan_follows_each_input(name, nan_input):
    """A NaN in one stored input at one time and cell makes the derived name
    NaN there only; ``ocean_sea_ice_fraction``, NaN on ice-free ocean in the
    stores, does not."""
    names = derived.DERIVED_INPUTS[name]
    tensors = {n: torch.rand(3, 2, 2, dtype=torch.float64) + 0.5 for n in names}
    tensors["sst"] = tensors.get("sst", torch.zeros(3, 2, 2)) + 270.0
    tensors["sea_surface_fraction"] = torch.ones(3, 2, 2, dtype=torch.float64)
    if "ocean_sea_ice_fraction" in tensors:
        tensors["ocean_sea_ice_fraction"][:, 1, 1] = float("nan")
    tensors[nan_input][2, 0, 1] = float("nan")
    out = derived.apply(tensors, [name], datetime.timedelta(days=5), {name})[name]
    expected_nan = torch.zeros(3, 2, 2, dtype=torch.bool)
    expected_nan[2, 0, 1] = True
    assert torch.equal(torch.isnan(out), expected_nan)


def test_derived_names_by_time_slice(tmp_path):
    """The inference path (``InferenceDataset._resolve_merged_datasets``) builds
    XarrayDatasets directly and reads windows with get_sample_by_time_slice."""
    config, joint = _write_stores(tmp_path)
    requested = ["sst", *DERIVED]
    schedule = IntSchedule(start_value=4, milestones=[])
    per_dataset_names = get_per_dataset_names(config, requested)
    dataset = MergedXarrayDataset(
        datasets=[
            XarrayDataset(c, n, schedule)
            for c, n in zip(config.merge, per_dataset_names)
        ],
        names=requested,
    )
    tensors, *_ = dataset.get_sample_by_time_slice(slice(5, 11))
    assert set(tensors) == set(requested)
    expected = _expected(joint, start=5, n=6)
    for name in DERIVED:
        torch.testing.assert_close(tensors[name], expected[name], equal_nan=True)
    assert (tensors["frozen_mass_energy_budget_residual"][0, _ICE] == 0).all()
    provider = dataset.properties.spatial_mask_provider
    for name in _ICE_MASKED:
        torch.testing.assert_close(
            provider.get_mask_tensor_for(name),
            torch.as_tensor(joint["mask_sea_ice_fraction"].values),
        )


def test_unrequested_inputs_are_dropped(tmp_path):
    config, _ = _write_stores(tmp_path)
    dataset, _ = _build(config, ["frozen_mass_energy_budget_residual"])
    tensors, *_ = dataset[0]
    assert set(tensors) == {"frozen_mass_energy_budget_residual"}


def test_no_derived_names_unchanged(tmp_path):
    config, joint = _write_stores(tmp_path)
    requested = ["sst", "DSWRFsfc", "simass", "sea_surface_fraction"]
    dataset, properties = _build(config, requested)
    tensors, *_ = dataset[2]
    assert set(tensors) == set(requested)
    torch.testing.assert_close(
        tensors["simass"], torch.as_tensor(joint["simass"].values[2:6]), equal_nan=True
    )
    for name in DERIVED:
        assert name not in properties.variable_metadata
    for props in (properties, dataset.properties):
        assert set(props.spatial_mask_provider.masks) == {
            "mask_2d",
            "mask_sea_ice_fraction",
        }


def test_missing_input_raises():
    tensors = {"simass": torch.zeros(2, 1, 1)}
    with pytest.raises(KeyError, match="sisnmass"):
        derived.apply(tensors, ["frozen_mass"], None, {"frozen_mass"})
