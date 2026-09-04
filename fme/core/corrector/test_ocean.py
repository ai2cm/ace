import dataclasses
import datetime
from typing import Any, cast

import dacite
import pytest
import torch

from fme import get_device
from fme.core.constants import DENSITY_OF_SEA_WATER_CM4, SPECIFIC_HEAT_OF_SEA_WATER_CM4
from fme.core.coordinates import DepthCoordinate
from fme.core.corrector.ocean import (
    OceanCorrectorConfig,
    OceanHeatContentBudgetConfig,
    OceanHeatContentCorrection,
    SeaIceFractionConfig,
    SurfaceEnergyFluxCorrectionConfig,
    _compute_ocean_net_surface_energy_flux,
)
from fme.core.gridded_ops import LatLonOperations
from fme.core.ocean_data import OceanData
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.typing_ import TensorDict, TensorMapping

DEVICE = get_device()
IMG_SHAPE = (5, 5)
NZ = 2

_MASK = torch.ones(*IMG_SHAPE, NZ, device=DEVICE)
_LAT, _LON = 2, 2
_MASK[_LAT, _LON, :] = 0.0


_MOCK_IDEPTH = torch.tensor([0.0, 5.0, 15.0], device=DEVICE)


class _MockDepth:
    def depth_integral(self, integrand: torch.Tensor) -> torch.Tensor:
        thickness = _MOCK_IDEPTH.diff(dim=-1)
        return torch.nansum(_MASK * integrand * thickness, dim=-1)

    @property
    def dz(self) -> torch.Tensor:
        return _MASK * _MOCK_IDEPTH.diff(dim=-1)


_VERTICAL_COORD = _MockDepth()


def test_ocean_corrector_force_positive():
    """"""
    torch.manual_seed(0)
    config = OceanCorrectorConfig(force_positive_names=["so_0", "so_1"])
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = config._build(ops, _VERTICAL_COORD, timestep)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    corrected_gen = corrector(input_data, gen_data, {}, None).corrected
    for name in ["so_0", "so_1"]:
        x = corrected_gen[name].clone()
        x[_LAT, _LON] = 0.0
        assert torch.all(x >= 0.0)


def test_sea_ice_fraction_keep_gradient_passes_gradient_through_clamp():
    config = SeaIceFractionConfig(
        sea_ice_fraction_name="sea_ice_fraction",
        land_fraction_name="land_fraction",
        remove_negative_ocean_fraction=False,
    )
    input_data = {"land_fraction": torch.zeros(IMG_SHAPE, device=DEVICE)}
    # values both below 0 and above 1 so the clamp saturates at both ends
    raw = torch.tensor([-0.5, 0.3, 1.5], device=DEVICE)

    sif_plain = raw.clone().requires_grad_(True)
    config({"sea_ice_fraction": sif_plain}, input_data)[
        "sea_ice_fraction"
    ].sum().backward()
    # plain clamp: zero gradient where saturated, one in the interior
    torch.testing.assert_close(
        sif_plain.grad, torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    )

    sif_ste = raw.clone().requires_grad_(True)
    out = config({"sea_ice_fraction": sif_ste}, input_data, keep_gradient=True)
    # forward value is still clamped to [0, 1]
    torch.testing.assert_close(
        out["sea_ice_fraction"], torch.tensor([0.0, 0.3, 1.0], device=DEVICE)
    )
    out["sea_ice_fraction"].sum().backward()
    torch.testing.assert_close(sif_ste.grad, torch.ones_like(raw))


def test_ocean_corrector_keep_gradient_through_clamps_forward_unchanged():
    # The straight-through flag must not change forward values; only gradients.
    torch.manual_seed(0)
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    sif = SeaIceFractionConfig(
        sea_ice_fraction_name="sea_ice_fraction",
        land_fraction_name="land_fraction",
    )
    input_data = {
        "land_fraction": torch.ones(IMG_SHAPE, device=DEVICE) * 0.3,
    }
    gen_data = {
        "so_0": torch.randn(IMG_SHAPE, device=DEVICE),
        "sea_ice_fraction": torch.randn(IMG_SHAPE, device=DEVICE),
    }
    baseline = (
        OceanCorrectorConfig(
            force_positive_names=["so_0"], sea_ice_fraction_correction=sif
        )
        ._build(ops, None, timestep)(input_data, gen_data, {}, None)
        .corrected
    )
    ste = (
        OceanCorrectorConfig(
            force_positive_names=["so_0"],
            sea_ice_fraction_correction=sif,
            keep_gradient_through_clamps=True,
        )
        ._build(ops, None, timestep)(input_data, gen_data, {}, None)
        .corrected
    )
    for name in baseline:
        torch.testing.assert_close(baseline[name], ste[name])


def test_ocean_corrector_has_no_negative_ocean_fraction():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    input_data["land_fraction"] = torch.ones(IMG_SHAPE, device=DEVICE) * 0.8
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data["sea_ice_fraction"] = torch.randn(IMG_SHAPE, device=DEVICE) * 0.5
    gen_data["sea_ice_fraction"][_LAT, _LON] = -0.5
    corrector = config._build(ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(
        input_data, gen_data, next_step_input_data, None
    ).corrected
    corrected_violation = (
        input_data["land_fraction"] + gen_data_corrected["sea_ice_fraction"]
    ) > 1.0
    assert not corrected_violation.any()
    assert not (gen_data_corrected["sea_ice_fraction"] < 0.0).any()


def test_ocean_corrector_has_negative_ocean_fraction():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            remove_negative_ocean_fraction=False,
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    input_data["land_fraction"] = torch.ones(IMG_SHAPE, device=DEVICE) * 0.8
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data["sea_ice_fraction"] = torch.randn(IMG_SHAPE, device=DEVICE) * 0.5
    gen_data["sea_ice_fraction"][_LAT, _LON] = -0.5
    corrector = config._build(ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(
        input_data, gen_data, next_step_input_data, None
    ).corrected
    corrected_violation = (
        input_data["land_fraction"] + gen_data_corrected["sea_ice_fraction"]
    ) > 1.0
    assert corrected_violation.any()
    # sea_ice_fraction values are still clamped to [0, 1]
    assert not (gen_data_corrected["sea_ice_fraction"] < 0.0).any()


def test_zero_where_ice_free_names():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["HI"],
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {"land_fraction": torch.ones(IMG_SHAPE, device=DEVICE)}
    input_data["land_fraction"][:3, :3] = torch.rand(3, 3, device=DEVICE)
    gen_data = {
        "sea_ice_fraction": torch.rand(IMG_SHAPE, device=DEVICE),
        "HI": torch.rand(IMG_SHAPE, device=DEVICE) * 10,
    }
    corrector = config._build(ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {}, None).corrected
    sea_ice_zero = gen_data_corrected["sea_ice_fraction"] == 0.0
    thickness = gen_data_corrected["HI"]
    torch.testing.assert_close(
        torch.where(sea_ice_zero, thickness, 0.0), torch.zeros_like(thickness)
    )


def test_zero_where_ice_free_names_multiple_variables():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["HI", "HS"],
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {"land_fraction": torch.ones(IMG_SHAPE, device=DEVICE)}
    input_data["land_fraction"][:3, :3] = torch.rand(3, 3, device=DEVICE)
    gen_data = {
        "sea_ice_fraction": torch.rand(IMG_SHAPE, device=DEVICE),
        "HI": torch.rand(IMG_SHAPE, device=DEVICE) * 10,
        "HS": torch.rand(IMG_SHAPE, device=DEVICE) * 5,
    }
    corrector = config._build(ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {}, None).corrected
    sea_ice_zero = gen_data_corrected["sea_ice_fraction"] == 0.0
    for name in ["HI", "HS"]:
        values = gen_data_corrected[name]
        torch.testing.assert_close(
            torch.where(sea_ice_zero, values, 0.0), torch.zeros_like(values)
        )


def test_from_state_migrates_sea_ice_thickness_name():
    state = {
        "sea_ice_fraction_correction": {
            "sea_ice_fraction_name": "ocean_sea_ice_fraction",
            "land_fraction_name": "land_fraction",
            "sea_ice_thickness_name": "HI",
            "remove_negative_ocean_fraction": False,
        },
    }
    config = OceanCorrectorConfig.from_state(state)
    assert config.sea_ice_fraction_correction is not None
    assert config.sea_ice_fraction_correction.zero_where_ice_free_names == ["HI"]


def test_from_state_migrates_sea_ice_thickness_name_none():
    state = {
        "sea_ice_fraction_correction": {
            "sea_ice_fraction_name": "ocean_sea_ice_fraction",
            "land_fraction_name": "land_fraction",
            "sea_ice_thickness_name": None,
            "remove_negative_ocean_fraction": False,
        },
    }
    config = OceanCorrectorConfig.from_state(state)
    assert config.sea_ice_fraction_correction is not None
    assert config.sea_ice_fraction_correction.zero_where_ice_free_names == []


def _make_atmos_forcing_data(shape, device=DEVICE):
    """Build atmosphere forcing tensors needed for the surface energy flux
    correction tests."""
    return {
        "DSWRFsfc": torch.full(shape, 200.0, device=device),
        "USWRFsfc": torch.full(shape, 50.0, device=device),
        "DLWRFsfc": torch.full(shape, 300.0, device=device),
        "ULWRFsfc": torch.full(shape, 350.0, device=device),
        "LHTFLsfc": torch.full(shape, 100.0, device=device),
        "SHTFLsfc": torch.full(shape, 20.0, device=device),
        "PRATEsfc": torch.full(shape, 1e-4, device=device),
        "total_frozen_precipitation_rate": torch.full(shape, 1e-5, device=device),
    }


def test_surface_energy_flux_correction_resid():
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="residual_prediction"
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = config._build(ops, None, timestep)

    sst = torch.full(IMG_SHAPE, 300.0, device=DEVICE)
    gen_hfds = torch.full(IMG_SHAPE, 5.0, device=DEVICE)
    sea_ice_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    sea_ice_fraction[0, :] = 0.3
    land_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    land_fraction[-1, :] = 1.0

    gen_data = {
        "sst": sst,
        "hfds": gen_hfds,
        "sea_ice_fraction": sea_ice_fraction,
    }
    forcing_data = {
        "land_fraction": land_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}

    ocean_fraction = 1 - land_fraction - sea_ice_fraction
    expected_net_flux = _compute_ocean_net_surface_energy_flux(input_data, sst)
    expected_hfds = gen_hfds + ocean_fraction * expected_net_flux

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
    torch.testing.assert_close(corrected["hfds"], expected_hfds)
    # on land ocean_fraction is 0, so hfds is unchanged
    torch.testing.assert_close(corrected["hfds"][-1, :], gen_hfds[-1, :])
    # with sea ice, correction is reduced relative to ice-free rows
    ice_row_correction = (corrected["hfds"][0, 0] - gen_hfds[0, 0]).abs()
    open_row_correction = (corrected["hfds"][1, 0] - gen_hfds[1, 0]).abs()
    assert ice_row_correction < open_row_correction


def test_surface_energy_flux_correction_prescribed():
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed"
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = config._build(ops, None, timestep)

    sst = torch.full(IMG_SHAPE, 300.0, device=DEVICE)
    gen_hfds = torch.full(IMG_SHAPE, 5.0, device=DEVICE)
    sea_ice_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    sea_ice_fraction[0, :] = 0.3
    land_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    land_fraction[-1, :] = 1.0

    gen_data = {
        "sst": sst,
        "hfds": gen_hfds,
        "sea_ice_fraction": sea_ice_fraction,
    }
    forcing_data = {
        "land_fraction": land_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}

    ocean_fraction = 1 - land_fraction - sea_ice_fraction
    net_flux = _compute_ocean_net_surface_energy_flux(input_data, sst)
    expected_hfds = net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
    torch.testing.assert_close(corrected["hfds"], expected_hfds)
    # on land (ocean_fraction=0), hfds equals gen_hfds
    torch.testing.assert_close(corrected["hfds"][-1, :], gen_hfds[-1, :])
    # in open ocean (no ice, no land), hfds equals net_flux
    open_ocean_row = 1
    torch.testing.assert_close(
        corrected["hfds"][open_ocean_row, :], net_flux[open_ocean_row, :]
    )


@pytest.mark.parametrize(
    "hfds_type",
    [
        pytest.param("input", id="hfds_in_input"),
        pytest.param("gen", id="hfds_in_gen"),
        pytest.param("total_area", id="hfds_total_area_in_gen"),
    ],
)
def test_ocean_heat_content_correction(hfds_type):
    config = OceanCorrectorConfig(
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="scaled_temperature",
            constant_unaccounted_heating=0.1,
        )
    )
    timestep = datetime.timedelta(seconds=5 * 24 * 3600)
    nsamples, nlat, nlon, nlevels = 4, 3, 3, 2
    mask = torch.ones(nsamples, nlat, nlon, nlevels)
    mask[:, 0, 0, 0] = 0.0
    mask[:, 0, 0, 1] = 0.0
    mask[:, 0, 1, 1] = 0.0
    masks = {
        "mask_0": mask[:, :, :, 0],
        "mask_1": mask[:, :, :, 1],
        "mask_2d": mask[:, :, :, 0],
    }
    spatial_mask_provider = SpatialMaskProvider(masks)
    ops = LatLonOperations(torch.ones(size=[3, 3]), spatial_mask_provider)

    idepth = torch.tensor([2.5, 10, 20])
    depth_coordinate = DepthCoordinate(idepth, mask)

    sea_surface_fraction = mask[:, :, :, 0]

    input_data_dict = {
        "thetao_0": torch.ones(nsamples, nlat, nlon),
        "thetao_1": torch.ones(nsamples, nlat, nlon),
        "sst": torch.ones(nsamples, nlat, nlon) + 273.15,
    }
    gen_data_dict = {
        "thetao_0": torch.ones(nsamples, nlat, nlon) * 2,
        "thetao_1": torch.ones(nsamples, nlat, nlon) * 2,
        "sst": torch.ones(nsamples, nlat, nlon) * 2 + 273.15,
    }
    if hfds_type == "gen":
        gen_data_dict["hfds"] = torch.ones(nsamples, nlat, nlon)
    elif hfds_type == "total_area":
        # hfds_total_area is already weighted by sea_surface_fraction also
        # include hfds with a different value to verify hfds_total_area takes
        # priority
        gen_data_dict["hfds"] = (
            torch.ones(nsamples, nlat, nlon) * 100
        )  # should be ignored
        gen_data_dict["hfds_total_area"] = (
            torch.ones(nsamples, nlat, nlon) * sea_surface_fraction
        )
    else:
        input_data_dict["hfds"] = torch.ones(nsamples, nlat, nlon)
    forcing_data_dict = {
        "hfgeou": torch.ones(nsamples, nlat, nlon),
        "sea_surface_fraction": sea_surface_fraction,
    }
    input_data = OceanData(input_data_dict, depth_coordinate)
    gen_data = OceanData(gen_data_dict, depth_coordinate)
    corrector = config._build(ops, depth_coordinate, timestep)
    result = corrector(input_data_dict, gen_data_dict, forcing_data_dict, None)
    gen_data_corrected_dict = result.corrected

    # the OHC correction writes every potential-temperature level and the SST;
    # the heat-flux fields are read but not written, so they stay out of the set
    assert set(result.modified_names) == {"thetao_0", "thetao_1", "sst"}
    for name, delta in result.diagnostics.delta.items():
        torch.testing.assert_close(
            delta, result.corrected[name] - gen_data_dict[name], equal_nan=True
        )

    input_ohc = input_data.ocean_heat_content.nanmean(dim=(-1, -2), keepdim=True)
    gen_ohc = gen_data.ocean_heat_content.nanmean(dim=(-1, -2), keepdim=True)
    torch.testing.assert_close(
        gen_ohc,
        input_ohc * 2,
        equal_nan=True,
    )
    ohc_change = (
        2.1 * timestep.total_seconds()
    )  # 2.1 because of hfds + hfgeou + unaccounted heating
    corrector_ratio = (input_ohc + ohc_change) / gen_ohc
    expected_gen_data_dict = {
        key: value * corrector_ratio if key.startswith("thetao") else value
        for key, value in gen_data_dict.items()
    }
    expected_gen_data_dict["sst"] = (
        gen_data_dict["sst"] - 273.15
    ) * corrector_ratio + 273.15

    torch.testing.assert_close(
        gen_data_corrected_dict["sst"],
        expected_gen_data_dict["sst"],
    )

    expected_gen_data = OceanData(expected_gen_data_dict, depth_coordinate)
    gen_data_corrected = OceanData(gen_data_corrected_dict, depth_coordinate)
    torch.testing.assert_close(
        expected_gen_data.ocean_heat_content,
        gen_data_corrected.ocean_heat_content,
        equal_nan=True,
    )


def test_ocean_corrector_config_fields_are_known():
    # Staleness guard: if a new corrector option is added to
    # OceanCorrectorConfig this fails, flagging that the corrector delta/
    # modified-return tests need to exercise it.
    expected = {
        "force_positive_names",
        "sea_ice_fraction_correction",
        "surface_energy_flux_correction",
        "ocean_heat_content_correction",
        "keep_gradient_through_clamps",
        "corrector_disabled_epochs",  # inherited epoch-scheduling field
    }
    actual = {f.name for f in dataclasses.fields(OceanCorrectorConfig)}
    assert actual == expected, (
        "OceanCorrectorConfig fields changed; update the corrector delta tests "
        f"to cover the new option(s): {actual ^ expected}"
    )


def test_ocean_corrector_delta_matches_modified_returns():
    torch.manual_seed(0)
    config = OceanCorrectorConfig(
        force_positive_names=["so_0"],
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["HI", "HS"],
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = config._build(ops, None, timestep)
    input_data = {"land_fraction": torch.rand(IMG_SHAPE, device=DEVICE)}
    gen_data = {
        "so_0": torch.randn(IMG_SHAPE, device=DEVICE),
        "so_1": torch.randn(IMG_SHAPE, device=DEVICE),  # uncorrected field
        "sea_ice_fraction": torch.rand(IMG_SHAPE, device=DEVICE),
        "HI": torch.rand(IMG_SHAPE, device=DEVICE) * 10,
        "HS": torch.rand(IMG_SHAPE, device=DEVICE) * 5,
    }
    result = corrector(input_data, gen_data, {}, None)
    # delta keys are exactly the corrector's modified names
    assert set(result.diagnostics.delta) == set(result.modified_names)
    for name, delta in result.diagnostics.delta.items():
        torch.testing.assert_close(delta, result.corrected[name] - gen_data[name])
    assert set(result.modified_names) == {"so_0", "sea_ice_fraction", "HI", "HS"}
    # the uncorrected field passes through unchanged and is absent from the set
    assert "so_1" not in result.modified_names
    torch.testing.assert_close(result.corrected["so_1"], gen_data["so_1"])


def test_ocean_corrector_empty_delta_when_nothing_modified():
    # A corrector with no field-modifying option emits an empty delta and an
    # unchanged copy of gen_data.
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = OceanCorrectorConfig()._build(ops, None, timestep)
    gen_data = {"so_0": torch.randn(IMG_SHAPE, device=DEVICE)}
    result = corrector({}, gen_data, {}, None)
    assert dict(result.diagnostics.delta) == {}
    assert set(result.modified_names) == set()
    torch.testing.assert_close(result.corrected["so_0"], gen_data["so_0"])


@pytest.mark.parametrize("method", ["scaled_temperature", "uniform_temperature"])
def test_ocean_corrector_is_per_member_under_ensemble_folding(method):
    """Ensemble training folds the ensemble members into the batch dimension, so
    the corrector sees several members at once. Every correction must act
    per-member: one that coupled across the batch dim (e.g. a global mean taken
    over samples too) would tie the members together and silently collapse the
    ensemble spread the proper scoring rule is meant to reward.
    """
    torch.manual_seed(0)
    n_members, nlat, nlon, nlevels = 2, 3, 3, 2
    config = OceanCorrectorConfig(
        force_positive_names=["so_0", "so_1"],
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["sea_ice_thickness"],
        ),
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method=method,
            constant_unaccounted_heating=0.1,
        ),
    )
    timestep = datetime.timedelta(seconds=5 * 24 * 3600)
    mask = torch.ones(nlat, nlon, nlevels)
    mask[0, 0, :] = 0.0
    masks = {
        "mask_0": mask[:, :, 0],
        "mask_1": mask[:, :, 1],
        "mask_2d": mask[:, :, 0],
    }
    # non-uniform in latitude only, as the area weights require
    area = torch.tensor([0.5, 1.0, 1.5]).unsqueeze(-1).expand(nlat, nlon)
    ops = LatLonOperations(area, SpatialMaskProvider(masks))
    depth_coordinate = DepthCoordinate(torch.tensor([2.5, 10.0, 20.0]), mask)
    corrector = config._build(ops, depth_coordinate, timestep)

    def randoms(shape):
        return torch.randn(shape)

    input_data = {
        "thetao_0": randoms((n_members, nlat, nlon)) + 2.0,
        "thetao_1": randoms((n_members, nlat, nlon)) + 2.0,
        "sst": randoms((n_members, nlat, nlon)) + 275.0,
        "land_fraction": torch.zeros(n_members, nlat, nlon),
    }
    # members differ in every generated field, as they would under different
    # noise draws
    gen_data = {
        "thetao_0": randoms((n_members, nlat, nlon)) + 2.0,
        "thetao_1": randoms((n_members, nlat, nlon)) + 2.0,
        "sst": randoms((n_members, nlat, nlon)) + 275.0,
        "so_0": randoms((n_members, nlat, nlon)),
        "so_1": randoms((n_members, nlat, nlon)),
        # spans the clamp range at both ends so the sea-ice rebalance engages
        "sea_ice_fraction": randoms((n_members, nlat, nlon)) * 0.8 + 0.5,
        "sea_ice_thickness": randoms((n_members, nlat, nlon)),
        "hfds": randoms((n_members, nlat, nlon)),
    }
    forcing_data = {
        "hfgeou": randoms((n_members, nlat, nlon)),
        "sea_surface_fraction": mask[:, :, 0].expand(n_members, nlat, nlon),
    }

    folded = corrector(input_data, gen_data, forcing_data, None).corrected
    assert set(folded) >= {"so_0", "sea_ice_fraction", "thetao_0", "sst"}

    for member in range(n_members):

        def slice_member(data, member=member):
            return {name: value[member : member + 1] for name, value in data.items()}

        alone = corrector(
            slice_member(input_data),
            slice_member(gen_data),
            slice_member(forcing_data),
            None,
        ).corrected
        for name, value in alone.items():
            torch.testing.assert_close(
                folded[name][member : member + 1],
                value,
                msg=lambda m, name=name, member=member: (
                    f"{name} for member {member} depends on the other members: {m}"
                ),
            )

    # and the members really are distinct after correction, so the comparison
    # above is not vacuous
    for name in folded:
        assert not torch.allclose(folded[name][0], folded[name][1])


_SEA_FLOOR_NLAT, _SEA_FLOOR_NLON, _SEA_FLOOR_NZ = 3, 3, 3
_SEA_FLOOR_IDEPTH = torch.tensor([0.0, 10.0, 30.0, 60.0], device=DEVICE)
# hfds + hfgeou in the fixture, uniform over the wet columns, in W/m**2
_SEA_FLOOR_NET_FLUX = 4.0
_SEA_FLOOR_TIMESTEP = datetime.timedelta(seconds=5 * 24 * 3600)


def _make_sea_floor_fixture(nsamples: int = 2, seed: int = 0):
    """Build a masked multi-level ocean fixture with a non-trivial sea floor.

    Three of the nine columns are special: one is all land, one has only its
    surface layer in the water, and one has two of its three layers. Every wet
    column's deepest valid layer is a partial bottom cell (``deptho`` falls
    inside it), so the effective thickness is neither the nominal layer
    thickness nor a 0/1 multiple of it, and a correction derived from a
    hand-rolled nominal ``dz`` sum would not conserve heat.

    Returns:
        ``(ops, depth_coordinate, input_data, gen_data, forcing_data)``.
    """
    torch.manual_seed(seed)
    nlat, nlon, nz = _SEA_FLOOR_NLAT, _SEA_FLOOR_NLON, _SEA_FLOOR_NZ
    n_valid_levels = torch.full((nlat, nlon), nz, device=DEVICE)
    n_valid_levels[0, 0] = 0  # all land
    n_valid_levels[0, 1] = 1  # only the surface layer is in the water
    n_valid_levels[1, 1] = 2
    levels = torch.arange(nz, device=DEVICE)
    mask = (levels < n_valid_levels.unsqueeze(-1)).to(torch.float32)
    deptho = torch.tensor(
        [[0.0, 7.0, 45.0], [45.0, 22.0, 45.0], [45.0, 45.0, 52.0]], device=DEVICE
    )
    depth_coordinate = DepthCoordinate(_SEA_FLOOR_IDEPTH, mask, deptho)
    masks: TensorDict = {f"mask_{k}": mask[:, :, k] for k in range(nz)}
    masks["mask_2d"] = mask[:, :, 0]
    # non-uniform in latitude only, as the area weights require
    area = torch.tensor([0.5, 1.0, 1.5], device=DEVICE).unsqueeze(-1).expand(nlat, nlon)
    ops = LatLonOperations(area, SpatialMaskProvider(masks))
    shape = (nsamples, nlat, nlon)

    def rand(offset: float) -> torch.Tensor:
        return torch.rand(shape, device=DEVICE) * 4.0 + offset

    input_data = {f"thetao_{k}": rand(1.0) for k in range(nz)}
    input_data["sst"] = rand(274.15)
    gen_data = {f"thetao_{k}": rand(1.0) for k in range(nz)}
    gen_data["sst"] = rand(274.15)
    gen_data["hfds"] = torch.full(shape, 3.0, device=DEVICE)
    forcing_data = {
        "hfgeou": torch.full(shape, 1.0, device=DEVICE),
        "sea_surface_fraction": mask[:, :, 0].expand(shape),
    }
    return ops, depth_coordinate, input_data, gen_data, forcing_data


def _build_ohc_corrector(ops, depth_coordinate, method, unaccounted_heating=0.0):
    return OceanCorrectorConfig(
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method=method,
            constant_unaccounted_heating=unaccounted_heating,
        )
    )._build(ops, depth_coordinate, _SEA_FLOOR_TIMESTEP)


def _global_mean_ohc(ops, depth_coordinate, data: TensorMapping) -> torch.Tensor:
    return ops.area_weighted_mean(
        OceanData(data, depth_coordinate).ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )


@pytest.mark.parametrize("unaccounted_heating", [0.0, 0.1])
def test_uniform_temperature_conserves_ocean_heat_content(unaccounted_heating):
    # The additive correction must hit the same budget the multiplicative one
    # does. This is what pins the increment's denominator: if it came from
    # anything other than depth_integral over the same columns as the heat
    # content (a nominal dz sum, or a mean including the dry columns) the
    # corrected heat content would miss the target.
    ops, depth_coordinate, input_data, gen_data, forcing_data = (
        _make_sea_floor_fixture()
    )
    corrector = _build_ohc_corrector(
        ops, depth_coordinate, "uniform_temperature", unaccounted_heating
    )
    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
    expected_change = (
        _SEA_FLOOR_NET_FLUX + unaccounted_heating
    ) * _SEA_FLOOR_TIMESTEP.total_seconds()
    target = _global_mean_ohc(ops, depth_coordinate, input_data) + expected_change
    torch.testing.assert_close(
        _global_mean_ohc(ops, depth_coordinate, corrected), target, rtol=1e-6, atol=0.0
    )


def test_uniform_temperature_deposits_heat_proportional_to_thickness():
    # The property the whole experiment turns on: uniform_temperature deposits
    # heat in proportion to dz_k, scaled_temperature in proportion to T_k * dz_k.
    ops, depth_coordinate, input_data, gen_data, forcing_data = (
        _make_sea_floor_fixture()
    )
    nz = _SEA_FLOOR_NZ
    dz = depth_coordinate.dz
    valid = dz > 0.0
    uniform = _build_ohc_corrector(ops, depth_coordinate, "uniform_temperature")(
        input_data, gen_data, forcing_data, None
    ).corrected
    scaled = _build_ohc_corrector(ops, depth_coordinate, "scaled_temperature")(
        input_data, gen_data, forcing_data, None
    ).corrected

    uniform_increment = [
        uniform[f"thetao_{k}"] - gen_data[f"thetao_{k}"] for k in range(nz)
    ]
    # one global increment per sample, read off a column where every level is
    # valid; the correction is per-sample, so keep the sample dimension
    delta_temperature = uniform_increment[0][:, 2:3, 2:3]
    heat_capacity = SPECIFIC_HEAT_OF_SEA_WATER_CM4 * DENSITY_OF_SEA_WATER_CM4
    for k in range(nz):
        expected = torch.where(
            valid[..., k], delta_temperature, torch.zeros_like(delta_temperature)
        ).expand(uniform_increment[k].shape)
        torch.testing.assert_close(uniform_increment[k], expected, rtol=1e-5, atol=1e-8)
        # so the heat added at each level is cp * rho * dz_k * delta_T: the only
        # k dependence is dz_k
        heat_added = heat_capacity * dz[..., k] * uniform_increment[k]
        torch.testing.assert_close(
            heat_added,
            heat_capacity * dz[..., k] * delta_temperature.expand(heat_added.shape),
            rtol=1e-5,
            atol=1e-8,
        )

    scaled_increment = [
        scaled[f"thetao_{k}"] - gen_data[f"thetao_{k}"] for k in range(nz)
    ]
    # contrast: the multiplicative increment is (ratio - 1) * T_k, so the heat
    # added at each level goes as T_k * dz_k
    ratio_minus_one = (
        scaled_increment[0][:, 2:3, 2:3] / gen_data["thetao_0"][:, 2:3, 2:3]
    )
    for k in range(nz):
        torch.testing.assert_close(
            scaled_increment[k],
            gen_data[f"thetao_{k}"] * ratio_minus_one,
            rtol=1e-5,
            atol=1e-8,
        )
    # and the two profiles are genuinely different on this fixture
    assert not torch.allclose(scaled_increment[1], uniform_increment[1])


def test_uniform_temperature_leaves_invalid_cells_unchanged():
    # Invalid cells hold raw denormalized network output, and with no spatial
    # mask provider nothing overwrites them after the corrector, so the
    # increment must not shift them. This is the assertion the dz > 0 mask
    # exists for; conservation holds with or without it.
    ops, depth_coordinate, input_data, gen_data, forcing_data = (
        _make_sea_floor_fixture()
    )
    corrected = _build_ohc_corrector(ops, depth_coordinate, "uniform_temperature")(
        input_data, gen_data, forcing_data, None
    ).corrected
    dz = depth_coordinate.dz
    for k in range(_SEA_FLOOR_NZ):
        name = f"thetao_{k}"
        invalid = (dz[..., k] == 0.0).expand(gen_data[name].shape)
        assert invalid.any(), f"fixture has no invalid cell at level {k}"
        torch.testing.assert_close(
            corrected[name][invalid], gen_data[name][invalid], rtol=0.0, atol=0.0
        )
    # the sst on a dry column is likewise untouched
    dry = (dz[..., 0] == 0.0).expand(gen_data["sst"].shape)
    torch.testing.assert_close(
        corrected["sst"][dry], gen_data["sst"][dry], rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("method", ["scaled_temperature", "uniform_temperature"])
def test_ocean_heat_content_correction_is_differentiable(method):
    # The correction runs inside the training loop and the loss differentiates
    # through it, so the corrected output must stay on the autograd graph.
    ops, depth_coordinate, input_data, gen_data, forcing_data = (
        _make_sea_floor_fixture()
    )
    network_output = {
        name: value.clone().requires_grad_(True) for name, value in gen_data.items()
    }
    corrected = _build_ohc_corrector(ops, depth_coordinate, method)(
        input_data, network_output, forcing_data, None
    ).corrected
    loss = corrected["sst"].sum()
    for k in range(_SEA_FLOOR_NZ):
        loss = loss + corrected[f"thetao_{k}"].sum()
    loss.backward()
    # the temperature levels and the surface heat flux the budget reads all
    # carry gradient
    for name in [f"thetao_{k}" for k in range(_SEA_FLOOR_NZ)] + ["hfds"]:
        grad = network_output[name].grad
        assert grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(grad).all(), f"non-finite gradient at {name}"
        assert (grad != 0.0).any(), f"gradient at {name} is identically zero"


def test_ocean_heat_content_method_round_trips_through_from_state():
    config = OceanCorrectorConfig.from_state(
        {
            "ocean_heat_content_correction": {
                "method": "uniform_temperature",
                "constant_unaccounted_heating": 0.25,
            }
        }
    )
    assert config.ocean_heat_content_correction == OceanHeatContentBudgetConfig(
        method="uniform_temperature", constant_unaccounted_heating=0.25
    )


def test_ocean_heat_content_unknown_method_raises():
    with pytest.raises(dacite.DaciteError):
        OceanCorrectorConfig.from_state(
            {"ocean_heat_content_correction": {"method": "not_a_method"}}
        )
    # a method that got past config validation still fails loudly at the write
    ops, depth_coordinate, input_data, gen_data, forcing_data = (
        _make_sea_floor_fixture()
    )
    correction = OceanHeatContentCorrection(
        area_weighted_mean=ops.area_weighted_mean,
        vertical_coordinate=depth_coordinate,
        timestep_seconds=_SEA_FLOOR_TIMESTEP.total_seconds(),
        method=cast(Any, "not_a_method"),
        unaccounted_heating=0.0,
    )
    with pytest.raises(NotImplementedError):
        correction(input_data, gen_data, forcing_data, None)
