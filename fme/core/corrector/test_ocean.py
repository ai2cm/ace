import dataclasses
import datetime

import pytest
import torch

from fme import get_device
from fme.core.constants import LATENT_HEAT_OF_FREEZING
from fme.core.coordinates import (
    DepthCoordinate,
    LatLonCoordinates,
    NullVerticalCoordinate,
)
from fme.core.corrector.ocean import (
    FrozenMassBudgetConfig,
    FrozenMassBudgetCorrection,
    OceanCorrectorConfig,
    OceanHeatContentBudgetConfig,
    SeaIceFractionConfig,
    SurfaceEnergyFluxCorrectionConfig,
    _compute_ocean_net_surface_energy_flux,
)
from fme.core.dataset import derived
from fme.core.dataset_info import DatasetInfo
from fme.core.frozen_mass_budget import frozen_mass_flux_sum
from fme.core.gridded_ops import LatLonOperations
from fme.core.ocean_data import OceanData
from fme.core.spatial_mask_provider import NullSpatialMaskProvider, SpatialMaskProvider
from fme.core.typing_ import TensorMapping

DEVICE = get_device()
IMG_SHAPE = (5, 5)
NZ = 2

_MASK = torch.ones(*IMG_SHAPE, NZ, device=DEVICE)
_LAT, _LON = 2, 2
_MASK[_LAT, _LON, :] = 0.0


class _MockDepth:
    def depth_integral(self, integrand: torch.Tensor) -> torch.Tensor:
        idepth = torch.tensor([0, 5, 15], device=DEVICE)
        thickness = idepth.diff(dim=-1)
        return torch.nansum(_MASK * integrand * thickness, dim=-1)


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
    "hfds_name",
    [
        pytest.param("hfds", id="hfds_in_gen"),
        pytest.param("hfds_total_area", id="hfds_total_area_in_gen"),
    ],
)
def test_surface_energy_flux_correction_prescribed_open_ocean(hfds_name):
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_open_ocean"
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
        hfds_name: gen_hfds,
        "sea_ice_fraction": sea_ice_fraction,
    }
    forcing_data = {
        "land_fraction": land_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}

    ocean_fraction = 1 - land_fraction - sea_ice_fraction
    net_flux = _compute_ocean_net_surface_energy_flux(input_data, sst)
    if hfds_name == "hfds_total_area":
        net_flux = net_flux * (1 - land_fraction)
    expected_hfds = torch.where(ocean_fraction == 1, net_flux, gen_hfds)

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
    torch.testing.assert_close(corrected[hfds_name], expected_hfds)
    # open ocean (ocean_fraction exactly 1): hfds is the prescribed net flux
    open_ocean_row = 1
    torch.testing.assert_close(
        corrected[hfds_name][open_ocean_row, :], net_flux[open_ocean_row, :]
    )
    # partial ocean under sea ice: gen_hfds passes through unweighted, unlike
    # the "prescribed" method which would blend it with the net flux
    torch.testing.assert_close(corrected[hfds_name][0, :], gen_hfds[0, :])
    # land: gen_hfds passes through
    torch.testing.assert_close(corrected[hfds_name][-1, :], gen_hfds[-1, :])


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
    mask = torch.ones(nlat, nlon, nlevels)
    mask[0, 0, 0] = 0.0
    mask[0, 0, 1] = 0.0
    mask[0, 1, 1] = 0.0
    masks = {
        "mask_0": mask[:, :, 0],
        "mask_1": mask[:, :, 1],
        "mask_2d": mask[:, :, 0],
    }
    spatial_mask_provider = SpatialMaskProvider(masks)
    ops = LatLonOperations(torch.ones(size=[3, 3]), spatial_mask_provider)

    idepth = torch.tensor([2.5, 10, 20])
    depth_coordinate = DepthCoordinate(idepth, mask)

    sea_surface_fraction = mask[:, :, 0]

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
        "frozen_mass_budget_correction",
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


def test_ocean_corrector_is_per_member_under_ensemble_folding():
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
            method="scaled_temperature",
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


def _runoff_calving_case(hfds_name: str):
    """Open ocean everywhere (no land, no ice) except one land row."""
    torch.manual_seed(0)
    sst = torch.full(IMG_SHAPE, 290.0, device=DEVICE)
    land_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    land_fraction[-1, :] = 1.0
    gen_data = {
        "sst": sst,
        hfds_name: torch.randn(IMG_SHAPE, device=DEVICE),
        "sea_ice_fraction": torch.zeros(IMG_SHAPE, device=DEVICE),
        "hfrunoffds": torch.rand(IMG_SHAPE, device=DEVICE) * 5.0,
        "calving_residue": torch.randn(IMG_SHAPE, device=DEVICE),
    }
    forcing_data = {
        "land_fraction": land_fraction,
        "sea_surface_fraction": 1 - land_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}
    return input_data, gen_data, forcing_data


def _surface_flux_corrector(runoff_and_calving: bool | None):
    kwargs = (
        {} if runoff_and_calving is None else {"runoff_and_calving": runoff_and_calving}
    )
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed", **kwargs
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    return config._build(ops, None, datetime.timedelta(seconds=3600))


@pytest.mark.parametrize("hfds_name", ["hfds", "hfds_total_area"])
def test_surface_energy_flux_correction_runoff_and_calving(hfds_name):
    input_data, gen_data, forcing_data = _runoff_calving_case(hfds_name)
    ssf = forcing_data["sea_surface_fraction"]
    net_flux = _compute_ocean_net_surface_energy_flux(input_data, gen_data["sst"])
    extra = gen_data["hfrunoffds"] - gen_data["calving_residue"]

    on = _surface_flux_corrector(True)(input_data, gen_data, forcing_data, None)
    off = _surface_flux_corrector(False)(input_data, gen_data, forcing_data, None)
    default = _surface_flux_corrector(None)(input_data, gen_data, forcing_data, None)

    open_ocean = slice(0, -1)
    if hfds_name == "hfds_total_area":
        expected_on = ssf * (net_flux + extra)
        expected_off = ssf * net_flux
    else:
        expected_on = net_flux + extra
        expected_off = net_flux
    torch.testing.assert_close(
        on.corrected[hfds_name][open_ocean], expected_on[open_ocean]
    )
    torch.testing.assert_close(
        off.corrected[hfds_name][open_ocean], expected_off[open_ocean]
    )
    # on land (ocean_fraction 0) the generated value passes through
    torch.testing.assert_close(on.corrected[hfds_name][-1], gen_data[hfds_name][-1])
    # default off: identical to the flag unset
    torch.testing.assert_close(default.corrected[hfds_name], off.corrected[hfds_name])
    assert set(on.modified_names) == {hfds_name}


@pytest.mark.parametrize("missing", ["hfrunoffds", "calving_residue"])
def test_surface_energy_flux_correction_runoff_and_calving_missing_raises(missing):
    input_data, gen_data, forcing_data = _runoff_calving_case("hfds_total_area")
    del gen_data[missing]
    with pytest.raises(ValueError, match=missing):
        _surface_flux_corrector(True)(input_data, gen_data, forcing_data, None)
    # the flag off does not need them
    _surface_flux_corrector(False)(input_data, gen_data, forcing_data, None)


_FM_LAT = torch.tensor([-60.0, -20.0, 0.0, 30.0, 70.0])
_FM_SHAPE = (2, 5, 4)  # (sample, lat, lon)
_FM_DT = 5 * 86400.0
_FM_AREA = torch.tensor([0.5, 1.0, 1.2, 1.0, 0.4]).unsqueeze(-1).expand(5, 4)
_FM_HEMISPHERES = (_FM_LAT >= 0, _FM_LAT < 0)


def _fm_corrector(floor_mass_per_fraction: float = 905.0 * 1.0e-10):
    config = OceanCorrectorConfig(
        frozen_mass_budget_correction=FrozenMassBudgetConfig(
            floor_mass_per_fraction=floor_mass_per_fraction
        ),
    )
    ops = LatLonOperations(_FM_AREA)
    return config._build(ops, None, datetime.timedelta(seconds=_FM_DT), lat_1d=_FM_LAT)


def _fm_sum(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Per-sample hemisphere total, shape (sample, 1, 1), float64."""
    w = (_FM_AREA * h.unsqueeze(-1)).to(torch.float64)
    return (x.to(torch.float64) * w).sum(dim=(-2, -1), keepdim=True)


def _fm_case(factors: list[list[float]], seed: int = 0):
    """Random step whose r_hat puts m_diag = factor_h(sample) * m_hat.

    ``factors[sample][i]`` for hemisphere i of ``_FM_HEMISPHERES``.
    """
    g = torch.Generator().manual_seed(seed)

    def r(scale=1.0, offset=0.0):
        return offset + scale * torch.rand(_FM_SHAPE, generator=g, dtype=torch.float64)

    ssf = torch.ones(_FM_SHAPE, dtype=torch.float64)
    ssf[:, 1, 1] = 0.3
    ssf[:, 3, 0] = 0.0
    sif = r()
    sif[:, 0, 0] = 0.0  # ice-free cells
    sif[:, 4, 3] = 0.0
    m_hat = r(1000.0) * (sif > 0) * (ssf > 0)  # 0 on the all-land cell
    m_hat[:, 3, 2] = 0.0
    forcing_data = {
        "sea_surface_fraction": ssf,
        **{
            k: v.to(torch.float64).expand(_FM_SHAPE)
            for k, v in _make_atmos_forcing_data(_FM_SHAPE[1:], "cpu").items()
        },
    }
    gen_data = {
        "frozen_mass": m_hat,
        "ocean_sea_ice_fraction": sif,
        "hfds_total_area": r(100.0, -50.0),
        "hfrunoffds": r(5.0),
        "calving_residue": r(20.0, -10.0),
    }
    input_data = {"frozen_mass": r(1000.0)}
    factor = torch.zeros(_FM_SHAPE, dtype=torch.float64)
    for sample, row in enumerate(factors):
        for h, fac in zip(_FM_HEMISPHERES, row):
            factor[sample, h] = fac
    m_diag = factor * m_hat
    flux_sum = frozen_mass_flux_sum(
        forcing_data,
        gen_data["hfds_total_area"],
        gen_data["hfrunoffds"],
        gen_data["calving_residue"],
    )
    gen_data["frozen_mass_energy_budget_residual"] = (
        flux_sum
        - LATENT_HEAT_OF_FREEZING * (m_diag - input_data["frozen_mass"]) / _FM_DT
    )
    return input_data, gen_data, forcing_data, m_diag, ssf * sif


def _assert_hemisphere_budget(m_c, m_diag):
    for h in _FM_HEMISPHERES:
        torch.testing.assert_close(
            _fm_sum(m_c, h), _fm_sum(m_diag, h).clamp(min=0), rtol=1e-10, atol=1e-6
        )
    assert (m_c >= 0).all()


def test_frozen_mass_budget_correction_grow():
    # sample 0 grows in both hemispheres, sample 1 grows by other factors
    input_data, gen_data, forcing_data, m_diag, f = _fm_case([[1.3, 1.1], [1.05, 2.0]])
    result = _fm_corrector()(input_data, gen_data, forcing_data, None)
    m_c = result.corrected["frozen_mass"]
    assert set(result.modified_names) == {"frozen_mass"}
    _assert_hemisphere_budget(m_c, m_diag)
    m_hat = gen_data["frozen_mass"]
    for h in _FM_HEMISPHERES:
        deficit = _fm_sum(m_diag - m_hat, h)
        expected = m_hat + deficit * f / _fm_sum(f, h)
        torch.testing.assert_close(m_c[..., h, :], expected[..., h, :])


def test_frozen_mass_budget_correction_melt():
    input_data, gen_data, forcing_data, m_diag, _ = _fm_case([[0.6, 0.9], [0.2, 0.99]])
    m_c = _fm_corrector()(input_data, gen_data, forcing_data, None).corrected[
        "frozen_mass"
    ]
    _assert_hemisphere_budget(m_c, m_diag)
    m_hat = gen_data["frozen_mass"]
    # c f is negligible: the melt branch is a uniform scaling per hemisphere
    for h in _FM_HEMISPHERES:
        ratio = _fm_sum(m_diag, h) / _fm_sum(m_hat, h)
        torch.testing.assert_close(
            m_c[..., h, :], (m_hat * ratio)[..., h, :], rtol=1e-6, atol=1e-6
        )


def test_frozen_mass_budget_correction_melt_past_floor():
    # a large floor so the remainder branch keeps a visible s = min(m_hat, c f)
    c = 300.0
    input_data, gen_data, forcing_data, m_diag, f = _fm_case([[0.1, 0.05], [-0.5, 0.2]])
    m_hat = gen_data["frozen_mass"]
    available = torch.relu(m_hat - c * f)
    remainder = m_hat - available
    for h in _FM_HEMISPHERES:  # the case is in the remainder branch
        assert (_fm_sum(m_diag - m_hat, h) < -_fm_sum(available, h)).all()
    m_c = _fm_corrector(c)(input_data, gen_data, forcing_data, None).corrected[
        "frozen_mass"
    ]
    _assert_hemisphere_budget(m_c, m_diag)
    for h in _FM_HEMISPHERES:
        scale = (_fm_sum(m_diag, h) / _fm_sum(remainder, h)).clamp(min=0)
        torch.testing.assert_close(
            m_c[..., h, :], (remainder * scale)[..., h, :], rtol=1e-8, atol=1e-8
        )
    # sample 1, northern hemisphere: <m_diag> < 0, all mass removed
    assert (m_c[1, _FM_HEMISPHERES[0]] == 0).all()


@pytest.mark.parametrize("factor_sign", [1.0, -1.0])
def test_frozen_mass_budget_correction_ice_free_hemisphere(factor_sign):
    input_data, gen_data, forcing_data, _, _ = _fm_case([[1.2, 1.0], [0.8, 1.0]])
    south = _FM_HEMISPHERES[1]
    gen_data["frozen_mass"][:, south] = 0.0
    gen_data["ocean_sea_ice_fraction"][:, south] = 0.0
    # a nonzero budget deficit of either sign in the ice-free hemisphere
    gen_data["frozen_mass_energy_budget_residual"][:, south] -= factor_sign * 50.0
    m_hat = gen_data["frozen_mass"].clone().requires_grad_(True)
    r_hat = gen_data["frozen_mass_energy_budget_residual"].clone().requires_grad_(True)
    sif = gen_data["ocean_sea_ice_fraction"].clone().requires_grad_(True)
    gen_data.update(
        frozen_mass=m_hat,
        frozen_mass_energy_budget_residual=r_hat,
        ocean_sea_ice_fraction=sif,
    )
    m_c = _fm_corrector()(input_data, gen_data, forcing_data, None).corrected[
        "frozen_mass"
    ]
    assert (m_c[:, south] == 0).all()
    assert torch.isfinite(m_c).all()
    m_c.sum().backward()
    for x in (m_hat, r_hat, sif):
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_frozen_mass_budget_correction_float32_gradients_finite():
    input_data, gen_data, forcing_data, _, _ = _fm_case([[0.7, 1.4], [1.1, 0.5]])
    gen32 = {k: v.to(torch.float32).requires_grad_(True) for k, v in gen_data.items()}
    m_c = _fm_corrector()(
        {k: v.float() for k, v in input_data.items()},
        gen32,
        {k: v.float() for k, v in forcing_data.items()},
        None,
    ).corrected["frozen_mass"]
    assert m_c.dtype == torch.float32
    m_c.pow(2).sum().backward()
    for name, x in gen32.items():
        assert x.grad is not None and torch.isfinite(x.grad).all(), name


def test_frozen_mass_budget_correction_needs_lat():
    config = OceanCorrectorConfig(
        frozen_mass_budget_correction=FrozenMassBudgetConfig()
    )
    ops = LatLonOperations(_FM_AREA)
    with pytest.raises(ValueError, match="lat_1d"):
        config._build(ops, None, datetime.timedelta(seconds=_FM_DT))


def test_frozen_mass_budget_correction_missing_flux_raises():
    input_data, gen_data, forcing_data, _, _ = _fm_case([[1.0, 1.0], [1.0, 1.0]])
    del gen_data["calving_residue"]
    with pytest.raises(ValueError, match="calving_residue"):
        _fm_corrector()(input_data, gen_data, forcing_data, None)


_FM_OCEAN_STORE = [
    "ocean_sea_ice_fraction",
    "sst",
    "simass",
    "sisnmass",
    "hflso",
    "evs",
    "prsn",
    "hfrunoffds",
    "hfds_total_area",
]
_FM_GEN_NAMES = [
    "frozen_mass",
    "frozen_mass_energy_budget_residual",
    "calving_residue",
    "hfrunoffds",
    "hfds_total_area",
    "ocean_sea_ice_fraction",
    "sst",
]


def _fm_target_step(nan_on_land: bool):
    """A target step ``0 -> 1`` of a window whose derived names come from the
    loader (``derived.apply``), with the full corrector of the train config
    (``prescribed``, ``runoff_and_calving``, floor). Cell ``[3, 0]`` is all
    land; with ``nan_on_land`` the ocean-store fields are NaN there, as in the
    stores. Returns input, gen and forcing data, the corrector and the
    all-land mask.
    """
    g = torch.Generator().manual_seed(1)
    n_time, shape = 2, _FM_SHAPE[1:]

    def r(scale=1.0, offset=0.0):
        return offset + scale * torch.rand(
            n_time, *shape, generator=g, dtype=torch.float64
        )

    land = torch.zeros(shape, dtype=torch.float64)
    land[1, 1] = 0.7
    land[3, 0] = 1.0
    all_land = land == 1.0
    land_t = land.expand(n_time, *shape).clone()
    window = {
        "land_fraction": land_t,
        "sea_surface_fraction": 1 - land_t,
        "ocean_sea_ice_fraction": r() * (r() > 0.3),
        "sst": r(10.0, 270.0),
        "simass": r(900.0),
        "sisnmass": r(300.0),
        "hflso": r(20.0, -10.0),
        "evs": r(1e-5),
        "prsn": r(1e-5),
        "hfrunoffds": r(5.0),
        "hfds_total_area": r(100.0, -50.0),
        **{
            k: v.to(torch.float64) * r(0.5, 0.75)
            for k, v in _make_atmos_forcing_data((n_time, *shape), "cpu").items()
        },
    }
    if nan_on_land:
        for name in _FM_OCEAN_STORE:
            window[name][:, all_land] = float("nan")
    derived_names = [
        "frozen_mass",
        "calving_residue",
        "frozen_mass_energy_budget_residual",
    ]
    data = derived.apply(
        window,
        derived_names,
        datetime.timedelta(seconds=_FM_DT),
        keep=set(window) | set(derived_names),
    )
    forcing_names = [
        "land_fraction",
        "sea_surface_fraction",
        *_make_atmos_forcing_data((1,), "cpu"),
    ]
    input_data = {k: v[0:1] for k, v in data.items()}
    gen_data = {k: data[k][1:2] for k in _FM_GEN_NAMES}
    forcing_data = {k: data[k][1:2] for k in forcing_names}
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed", runoff_and_calving=True
        ),
        frozen_mass_budget_correction=FrozenMassBudgetConfig(),
    )
    corrector = config._build(
        LatLonOperations(_FM_AREA),
        None,
        datetime.timedelta(seconds=_FM_DT),
        lat_1d=_FM_LAT,
    )
    return input_data, gen_data, forcing_data, corrector, all_land


def test_frozen_mass_budget_correction_target_identity():
    """16-c03: the loader residual of a target step, fed as r_hat with the
    target frozen mass, returns the target frozen mass; the surface energy
    flux correction runs first, as in the train config."""
    input_data, gen_data, forcing_data, corrector, _ = _fm_target_step(False)
    result = corrector(input_data, gen_data, forcing_data, None)
    assert set(result.modified_names) == {"hfds_total_area", "frozen_mass"}
    torch.testing.assert_close(
        result.corrected["frozen_mass"], gen_data["frozen_mass"], rtol=1e-10, atol=1e-8
    )


@pytest.mark.parametrize("input_on_land", ["nan", "zero"])
def test_frozen_mass_budget_correction_target_identity_nan_on_land(input_on_land):
    """03a-c01: the 16-c03 identity holds on ocean cells when the target is
    NaN on land, with the input NaN there (no input masking) or 0 (the
    train config's ``input_masking``); land passes through."""
    input_data, gen_data, forcing_data, corrector, land = _fm_target_step(True)
    assert torch.isnan(gen_data["frozen_mass"][:, land]).all()
    if input_on_land == "zero":
        input_data = {k: torch.nan_to_num(v) for k, v in input_data.items()}
    m_c = corrector(input_data, gen_data, forcing_data, None).corrected["frozen_mass"]
    torch.testing.assert_close(
        m_c[:, ~land], gen_data["frozen_mass"][:, ~land], rtol=1e-10, atol=1e-8
    )
    assert torch.isnan(m_c[:, land]).all()


def test_frozen_mass_budget_correction_ignores_gen_on_land():
    """03a-c01: output masking runs after the corrector, so the gen values on
    land are network output; they do not change the ocean cells, and land
    passes through."""
    input_data, gen_data, forcing_data, corrector, land = _fm_target_step(True)
    input_data = {k: torch.nan_to_num(v) for k, v in input_data.items()}
    g = torch.Generator().manual_seed(2)
    noisy = {}
    for k, v in gen_data.items():
        v = v.clone()
        v[:, land] = 100.0 * torch.randn(v[:, land].shape, generator=g, dtype=v.dtype)
        noisy[k] = v
    m_c = corrector(input_data, noisy, forcing_data, None).corrected["frozen_mass"]
    torch.testing.assert_close(
        m_c[:, ~land], gen_data["frozen_mass"][:, ~land], rtol=1e-10, atol=1e-8
    )
    torch.testing.assert_close(m_c[:, land], noisy["frozen_mass"][:, land])


def _fm_mask() -> torch.Tensor:
    """``mask_frozen_mass`` for ``_fm_case``: 0 on the all-land cell and on
    ice-covered ocean cells Z, 1 elsewhere."""
    mask = torch.ones(_FM_SHAPE[1:], dtype=torch.float32)
    mask[3, 0] = 0.0
    mask[0, 2] = 0.0
    mask[2, 1] = 0.0
    mask[4, 0] = 0.0
    return mask


def _fm_masked_corrector(mask, floor_mass_per_fraction: float = 905.0 * 1.0e-10):
    config = OceanCorrectorConfig(
        frozen_mass_budget_correction=FrozenMassBudgetConfig(
            floor_mass_per_fraction=floor_mass_per_fraction
        ),
    )
    return config._build(
        LatLonOperations(_FM_AREA),
        None,
        datetime.timedelta(seconds=_FM_DT),
        lat_1d=_FM_LAT,
        frozen_mass_mask=mask,
    )


@pytest.mark.parametrize(
    "factors", [[[1.3, 1.1], [1.05, 2.0]], [[0.6, 0.9], [0.2, 0.99]]]
)
def test_frozen_mass_budget_correction_mask_budget_on_s(factors):
    """09: the hemisphere budget closes over S = wet & mask == 1, and
    frozen_mass passes through on Z = wet & mask == 0."""
    input_data, gen_data, forcing_data, m_diag, _ = _fm_case(factors)
    mask = _fm_mask()
    s = (forcing_data["sea_surface_fraction"] > 0) & (mask == 1)
    z = (forcing_data["sea_surface_fraction"] > 0) & (mask == 0)
    assert (gen_data["frozen_mass"][z] > 0).all()
    m_c = _fm_masked_corrector(mask)(
        input_data, gen_data, forcing_data, None
    ).corrected["frozen_mass"]
    for h in _FM_HEMISPHERES:
        torch.testing.assert_close(
            _fm_sum(m_c * s, h),
            _fm_sum(m_diag * s, h).clamp(min=0),
            rtol=1e-10,
            atol=1e-6,
        )
    assert (m_c[s] >= 0).all()
    assert torch.equal(m_c[z], gen_data["frozen_mass"][z])


def test_frozen_mass_budget_correction_mask_ignores_z():
    """09: values on Z of every field the corrector reads do not change m_c on
    S; m_c on Z is the (changed) gen frozen_mass."""
    input_data, gen_data, forcing_data, _, _ = _fm_case([[1.3, 0.7], [0.4, 1.6]])
    mask = _fm_mask()
    corrector = _fm_masked_corrector(mask)
    s = (forcing_data["sea_surface_fraction"] > 0) & (mask == 1)
    z = (forcing_data["sea_surface_fraction"] > 0) & (mask == 0)
    m_c = corrector(input_data, gen_data, forcing_data, None).corrected["frozen_mass"]
    g = torch.Generator().manual_seed(3)

    def perturb(data):
        out = {}
        for k, v in data.items():
            v = v.clone()
            v[z] = 100.0 * torch.randn(v[z].shape, generator=g, dtype=v.dtype)
            out[k] = v
        return out

    noisy_gen = perturb(gen_data)
    noisy_input = perturb(input_data)
    m_c_noisy = corrector(noisy_input, noisy_gen, forcing_data, None).corrected[
        "frozen_mass"
    ]
    torch.testing.assert_close(m_c_noisy[s], m_c[s], rtol=1e-12, atol=1e-9)
    assert torch.equal(m_c_noisy[z], noisy_gen["frozen_mass"][z])


def _fm_dataset_info(spatial_mask_provider):
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=_FM_LAT, lon=torch.tensor([0.0, 90.0, 180.0, 270.0])
        ),
        vertical_coordinate=NullVerticalCoordinate(),
        spatial_mask_provider=spatial_mask_provider,
        timestep=datetime.timedelta(seconds=_FM_DT),
    )


def _fm_get_corrector_result(dataset_info, mask):
    """``m_c`` from ``_get_corrector(dataset_info)`` and from ``_build`` with
    ``frozen_mass_mask=mask`` on the same grid."""
    config = OceanCorrectorConfig(
        frozen_mass_budget_correction=FrozenMassBudgetConfig()
    )
    input_data, gen_data, forcing_data, _, _ = _fm_case([[1.3, 0.7], [0.4, 1.6]])
    got = config._get_corrector(dataset_info)(
        input_data, gen_data, forcing_data, None
    ).corrected["frozen_mass"]
    expected = config._build(
        dataset_info.gridded_operations,
        None,
        datetime.timedelta(seconds=_FM_DT),
        lat_1d=_FM_LAT,
        frozen_mass_mask=mask,
    )(input_data, gen_data, forcing_data, None).corrected["frozen_mass"]
    return got, expected


@pytest.mark.parametrize("mask_name", ["mask_frozen_mass", "mask_2d"])
def test_frozen_mass_budget_correction_get_corrector_uses_mask(mask_name):
    """09: _get_corrector passes the provider's frozen_mass mask (or its
    mask_2d fallback) to the corrector."""
    mask = _fm_mask()
    dataset_info = _fm_dataset_info(SpatialMaskProvider({mask_name: mask}))
    config = OceanCorrectorConfig(
        frozen_mass_budget_correction=FrozenMassBudgetConfig()
    )
    (correction,) = config._get_corrector(dataset_info)._corrections
    assert isinstance(correction, FrozenMassBudgetCorrection)
    assert correction.mask is not None
    torch.testing.assert_close(correction.mask.cpu(), mask)
    got, expected = _fm_get_corrector_result(dataset_info, mask)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    unmasked = _fm_get_corrector_result(dataset_info, None)[1]
    assert not torch.equal(got, unmasked)


@pytest.mark.parametrize(
    "spatial_mask_provider",
    [
        None,
        NullSpatialMaskProvider,
        SpatialMaskProvider({"mask_sst": _fm_mask()}),
    ],
    ids=["missing", "null", "no_frozen_mass_mask"],
)
def test_frozen_mass_budget_correction_get_corrector_no_mask(spatial_mask_provider):
    """09: with no frozen_mass mask the corrector sums over wet, as before."""
    dataset_info = _fm_dataset_info(spatial_mask_provider)
    config = OceanCorrectorConfig(
        frozen_mass_budget_correction=FrozenMassBudgetConfig()
    )
    (correction,) = config._get_corrector(dataset_info)._corrections
    assert isinstance(correction, FrozenMassBudgetCorrection)
    assert correction.mask is None
    got, expected = _fm_get_corrector_result(dataset_info, None)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_frozen_mass_budget_correction_get_corrector_uses_frozen_mass_name():
    """09-c01: _get_corrector looks the mask up by
    FrozenMassBudgetConfig.frozen_mass_name, not the literal "frozen_mass"."""
    mask = _fm_mask()
    dataset_info = _fm_dataset_info(SpatialMaskProvider({"mask_ice_mass": mask}))
    config = OceanCorrectorConfig(
        frozen_mass_budget_correction=FrozenMassBudgetConfig(
            frozen_mass_name="ice_mass"
        )
    )
    (correction,) = config._get_corrector(dataset_info)._corrections
    assert isinstance(correction, FrozenMassBudgetCorrection)
    assert correction.mask is not None
    torch.testing.assert_close(correction.mask.cpu(), mask)

    def rename(data):
        return {("ice_mass" if k == "frozen_mass" else k): v for k, v in data.items()}

    input_data, gen_data, forcing_data, _, _ = _fm_case([[1.3, 0.7], [0.4, 1.6]])
    got = config._get_corrector(dataset_info)(
        rename(input_data), rename(gen_data), forcing_data, None
    ).corrected["ice_mass"]
    expected = _fm_get_corrector_result(dataset_info, mask)[1]
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
