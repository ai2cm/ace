import dataclasses
import datetime
from typing import NamedTuple

import pytest
import torch

from fme import get_device
from fme.core.constants import (
    DENSITY_OF_SEA_WATER_CM4,
    EARTH_RADIUS,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
)
from fme.core.coordinates import DepthCoordinate, LatLonCoordinates
from fme.core.corrector.ocean import (
    OceanCorrectorConfig,
    OceanHeatContentBudgetConfig,
    OceanSaltContentBudgetConfig,
    SeaIceFractionConfig,
    SeaSurfaceHeightSaltBudget,
    SeaSurfaceHeightSaltBudgetConfig,
    SurfaceEnergyFluxCorrectionConfig,
    ZosGlobalMeanCorrectionConfig,
    _compute_ocean_net_surface_energy_flux,
)
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import LatLonOperations
from fme.core.ocean_data import OceanData
from fme.core.spatial_mask_provider import SpatialMaskProvider
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


def test_ocean_net_surface_energy_flux_counts_snow_once():
    """PRATEsfc is total (liquid + frozen) precipitation."""
    cp, t_f, l_v, l_f = 3992.0, 273.15, 2.5e6, 334000.0
    am4 = _make_atmos_forcing_data((2, 2), device="cpu")
    sst = torch.full((2, 2), 300.0)
    f_top = (
        am4["DSWRFsfc"]
        - am4["USWRFsfc"]
        + am4["DLWRFsfc"]
        - am4["ULWRFsfc"]
        - am4["LHTFLsfc"]
        - am4["SHTFLsfc"]
    )
    p_h = cp * (am4["PRATEsfc"] - am4["LHTFLsfc"] / l_v) * (sst - t_f)
    torch.testing.assert_close(
        _compute_ocean_net_surface_energy_flux(am4, sst),
        f_top - l_f * am4["total_frozen_precipitation_rate"] + p_h,
    )


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


class _PartialLandOHCBudgetCase(NamedTuple):
    ops: LatLonOperations
    depth_coordinate: DepthCoordinate
    input_data: dict[str, torch.Tensor]
    gen_data: dict[str, torch.Tensor]
    forcing_data: dict[str, torch.Tensor]
    area: torch.Tensor
    sea_surface_fraction: torch.Tensor
    net_energy_flux_total_area: torch.Tensor


def _partial_land_ohc_budget_case(hfds_type: str) -> _PartialLandOHCBudgetCase:
    """Grid with land, partial-land and open-ocean columns."""
    torch.manual_seed(0)
    nlat, nlon, nlev = 4, 8, 3
    s = torch.ones(nlat, nlon, dtype=torch.float64)
    s[:, 0] = 0.0
    s[:, 1] = 0.35
    s[:, 2] = 0.8
    mask_2d = (s > 0).to(s.dtype)
    area = torch.cos(torch.linspace(-1.0, 1.0, nlat, dtype=torch.float64))
    area = area.unsqueeze(-1).expand(nlat, nlon).contiguous()
    ops = LatLonOperations(area, SpatialMaskProvider({"mask_2d": mask_2d}))
    depth_coordinate = DepthCoordinate(
        torch.tensor([0.0, 10.0, 100.0, 1000.0], dtype=torch.float64),
        mask_2d.unsqueeze(-1).expand(nlat, nlon, nlev).contiguous(),
    )
    base = torch.tensor([18.0, 12.0, 4.0], dtype=torch.float64)
    thetao_in = base + 0.5 * torch.randn(nlat, nlon, nlev, dtype=torch.float64)
    thetao_gen = thetao_in + 0.05 * torch.randn(nlat, nlon, nlev, dtype=torch.float64)
    hfds = 40.0 * torch.randn(nlat, nlon, dtype=torch.float64) * mask_2d
    hfgeou = 0.08 * mask_2d
    input_data = {f"thetao_{k}": thetao_in[..., k] for k in range(nlev)}
    gen_data = {f"thetao_{k}": thetao_gen[..., k] for k in range(nlev)}
    if hfds_type == "total_area":
        gen_data["hfds_total_area"] = hfds * s
    elif hfds_type == "gen":
        gen_data["hfds"] = hfds
    else:
        input_data["hfds"] = hfds
    forcing_data = {"hfgeou": hfgeou, "sea_surface_fraction": s}
    flux = (hfds + hfgeou) * s
    return _PartialLandOHCBudgetCase(
        ops=ops,
        depth_coordinate=depth_coordinate,
        input_data=input_data,
        gen_data=gen_data,
        forcing_data=forcing_data,
        area=area,
        sea_surface_fraction=s,
        net_energy_flux_total_area=flux,
    )


@pytest.mark.parametrize("hfds_type", ["total_area", "gen", "input"])
def test_ocean_heat_content_correction_partial_land_budget(hfds_type: str):
    """The corrected state closes sum(a s H_out) = sum(a s H_in) + sum(a F) dt,
    H per unit ocean area and F per unit total cell area.
    """
    case = _partial_land_ohc_budget_case(hfds_type)
    depth = case.depth_coordinate
    input_data, gen_data = case.input_data, case.gen_data
    area, s = case.area, case.sea_surface_fraction
    flux = case.net_energy_flux_total_area
    dt = 5 * 86400.0
    config = OceanCorrectorConfig(
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="scaled_temperature",
        )
    )
    corrector = config._build(case.ops, depth, datetime.timedelta(seconds=dt))
    corrected = corrector(input_data, gen_data, case.forcing_data, None).corrected

    def integral(field: torch.Tensor) -> torch.Tensor:
        return torch.nansum(area * field)

    h_in = OceanData(input_data, depth).ocean_heat_content
    h_out = OceanData(corrected, depth).ocean_heat_content
    torch.testing.assert_close(
        integral(s * h_out),
        integral(s * h_in) + integral(flux) * dt,
        rtol=1e-10,
        atol=0.0,
    )

    # closed-form ratio, independent of OceanData
    cprho = SPECIFIC_HEAT_OF_SEA_WATER_CM4 * DENSITY_OF_SEA_WATER_CM4
    dz = torch.tensor([10.0, 90.0, 900.0], dtype=torch.float64)
    thetao_in = torch.stack([input_data[f"thetao_{k}"] for k in range(3)], -1)
    thetao_gen = torch.stack([gen_data[f"thetao_{k}"] for k in range(3)], -1)
    w = area * (s > 0)
    H_in = (thetao_in * cprho * dz).sum(-1)
    H_gen = (thetao_gen * cprho * dz).sum(-1)
    r_expected = ((w * s * H_in).sum() + (w * flux).sum() * dt) / (w * s * H_gen).sum()
    wet = s > 0
    torch.testing.assert_close(
        corrected["thetao_0"][wet] / gen_data["thetao_0"][wet],
        r_expected.expand(int(wet.sum())),
        rtol=1e-12,
        atol=0.0,
    )


def _salt_coordinates(nlat: int, nlon: int) -> LatLonCoordinates:
    return LatLonCoordinates(
        lat=torch.linspace(-80.0, 80.0, nlat), lon=torch.arange(nlon) * 360.0 / nlon
    )


def _salt_dataset_info(
    ocean_mask: torch.Tensor, layer_thickness: tuple[float, float]
) -> DatasetInfo:
    """Lat-lon grid with non-uniform cell areas and a two-layer depth
    coordinate."""
    nlat, nlon = ocean_mask.shape
    masks = {"mask_0": ocean_mask, "mask_1": ocean_mask, "mask_2d": ocean_mask}
    # on the device, as the dataset properties are in a real run
    idepth = torch.tensor(
        [0.0, layer_thickness[0], sum(layer_thickness)], device=DEVICE
    )
    return DatasetInfo(
        horizontal_coordinates=_salt_coordinates(nlat, nlon),
        vertical_coordinate=DepthCoordinate(
            idepth, torch.stack([ocean_mask, ocean_mask], dim=-1).to(DEVICE)
        ),
        spatial_mask_provider=SpatialMaskProvider(masks),
        timestep=datetime.timedelta(seconds=5 * 24 * 3600),
    )


def _ocean_cell_area_m2(ocean_mask: torch.Tensor) -> torch.Tensor:
    """float64 cell areas from the same weights the corrector uses, with 0
    over land."""
    area_weights = _salt_coordinates(*ocean_mask.shape).area_weights
    cell_area = area_weights.to(DEVICE, torch.float64) * 4 * torch.pi * EARTH_RADIUS**2
    return cell_area * (ocean_mask.to(DEVICE) > 0)


def _total_salt_content(
    data: TensorMapping,
    ocean_cell_area: torch.Tensor,
    layer_thickness: tuple[float, float],
) -> torch.Tensor:
    """float64 reference total salt content in psu m**3 over ocean cells."""
    column = (
        data["so_0"].double() * layer_thickness[0]
        + data["so_1"].double() * layer_thickness[1]
    )
    return (column.nan_to_num() * ocean_cell_area).sum(dim=(-2, -1))


def _salt_ocean_mask() -> torch.Tensor:
    ocean_mask = torch.ones(4, 8)
    ocean_mask[1, 2] = 0.0  # a land cell
    return ocean_mask


@pytest.mark.parametrize("weight_by_sea_surface_fraction", [True, False])
def test_ocean_salt_content_correction_without_budget(weight_by_sea_surface_fraction):
    # With no budget the salt content changes only by the constant term. With
    # weight_by_sea_surface_fraction, the content of a partly-land cell counts
    # in proportion to its ocean part and the constant applies over the sea
    # surface area; otherwise both use the whole ocean cell area.
    torch.manual_seed(0)
    ocean_mask = _salt_ocean_mask()
    nlat, nlon = ocean_mask.shape
    layer_thickness = (10.0, 20.0)
    dataset_info = _salt_dataset_info(ocean_mask, layer_thickness)
    sea_surface_fraction = torch.rand(nlat, nlon, dtype=torch.float64) * ocean_mask
    sea_surface_fraction[0, :] = 1.0  # some wholly-ocean cells too
    constant = 3e-9
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=OceanSaltContentBudgetConfig(
            method="scaled_salinity",
            constant_unaccounted_salting=constant,
            weight_by_sea_surface_fraction=weight_by_sea_surface_fraction,
        )
    )
    corrector = config.get_corrector(dataset_info)

    def salinity(value):
        so = value + torch.rand(2, nlat, nlon, dtype=torch.float64, device=DEVICE)
        return so.where(ocean_mask.to(DEVICE) > 0, float("nan"))

    input_so, gen_so = salinity(34.0), salinity(35.0)
    input_data = {"so_0": input_so[0], "so_1": input_so[1]}
    gen_data = {"so_0": gen_so[0], "so_1": gen_so[1]}
    forcing_data = {"sea_surface_fraction": sea_surface_fraction.to(DEVICE)}
    result = corrector(input_data, gen_data, forcing_data, None)
    corrected = result.corrected

    assert set(result.modified_names) == {"so_0", "so_1"}
    area = _ocean_cell_area_m2(ocean_mask)
    if weight_by_sea_surface_fraction:
        area = area * forcing_data["sea_surface_fraction"]
    expected_change = (
        constant * dataset_info.timestep.total_seconds() * float(area.sum())
    )
    torch.testing.assert_close(
        _total_salt_content(corrected, area, layer_thickness),
        _total_salt_content(input_data, area, layer_thickness) + expected_change,
        rtol=1e-12,
        atol=0.0,
    )
    # by one ratio applied to every level
    ratio = corrected["so_0"] / gen_data["so_0"]
    torch.testing.assert_close(
        corrected["so_1"], gen_data["so_1"] * ratio, equal_nan=True
    )


def _ssh_budget_corrector(
    dataset_info: DatasetInfo, reference_salinity_psu: float = 35.0
) -> CorrectorABC:
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=OceanSaltContentBudgetConfig(
            method="scaled_salinity",
            budget_config=SeaSurfaceHeightSaltBudgetConfig(
                reference_salinity_psu=reference_salinity_psu,
            ),
        )
    )
    return config.get_corrector(dataset_info)


def _ssh_salt_states(
    ocean_mask: torch.Tensor, input_ssh: torch.Tensor, gen_ssh: torch.Tensor
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """float64 input and generated states with random salinity and the given
    sea surface heights, all NaN over land as in the data."""
    ocean = ocean_mask.to(DEVICE) > 0

    def state(so_value: float, ssh: torch.Tensor) -> dict[str, torch.Tensor]:
        so = so_value + torch.rand(
            2, *ocean_mask.shape, dtype=torch.float64, device=DEVICE
        )
        so = so.where(ocean, float("nan"))
        ssh = ssh.to(DEVICE, torch.float64).where(ocean, float("nan"))
        return {"so_0": so[0], "so_1": so[1], "SSH": ssh}

    return state(34.0, input_ssh), state(35.0, gen_ssh)


def test_ocean_salt_content_correction_sea_surface_height_budget():
    # The water the model adds as sea surface height dilutes the salt at the
    # reference salinity, with both the water and the content weighted by the
    # sea surface fraction read from the forcing data.
    torch.manual_seed(0)
    ocean_mask = _salt_ocean_mask()
    nlat, nlon = ocean_mask.shape
    layer_thickness = (10.0, 20.0)
    reference_salinity = 34.5
    corrector = _ssh_budget_corrector(
        _salt_dataset_info(ocean_mask, layer_thickness), reference_salinity
    )
    sea_surface_fraction = torch.rand(nlat, nlon, dtype=torch.float64) * ocean_mask
    sea_surface_fraction[0, :] = 1.0  # some wholly-ocean cells too
    input_ssh = 0.5 * torch.randn(nlat, nlon, dtype=torch.float64)
    gen_ssh = input_ssh + 1e-3 * torch.randn(nlat, nlon, dtype=torch.float64)
    input_data, gen_data = _ssh_salt_states(ocean_mask, input_ssh, gen_ssh)
    forcing_data = {"sea_surface_fraction": sea_surface_fraction.to(DEVICE)}
    result = corrector(input_data, gen_data, forcing_data, None)
    corrected = result.corrected

    # SSH is read but not written
    assert set(result.modified_names) == {"so_0", "so_1"}
    sea_surface_area = (
        _ocean_cell_area_m2(ocean_mask) * forcing_data["sea_surface_fraction"]
    )
    ssh_change = (gen_data["SSH"] - input_data["SSH"]).nan_to_num()
    expected_change = -reference_salinity * float((ssh_change * sea_surface_area).sum())
    assert expected_change != 0.0
    torch.testing.assert_close(
        _total_salt_content(corrected, sea_surface_area, layer_thickness),
        _total_salt_content(input_data, sea_surface_area, layer_thickness)
        + expected_change,
        rtol=1e-12,
        atol=0.0,
    )
    # by one ratio applied to every level
    ratio = corrected["so_0"] / gen_data["so_0"]
    torch.testing.assert_close(
        corrected["so_1"], gen_data["so_1"] * ratio, equal_nan=True
    )


@pytest.mark.parametrize("rise_m", [0.0, 2e-3])
def test_ocean_salt_content_correction_uniform_sea_surface_height_rise(rise_m):
    # A uniform rise of h over an all-ocean grid adds h times the ocean area of
    # water, which takes S_ref times that volume of salt content away; no rise
    # holds the content fixed.
    torch.manual_seed(0)
    nlat, nlon = 4, 8
    ocean_mask = torch.ones(nlat, nlon)
    layer_thickness = (10.0, 20.0)
    reference_salinity = 35.0
    corrector = _ssh_budget_corrector(
        _salt_dataset_info(ocean_mask, layer_thickness), reference_salinity
    )
    input_ssh = 0.5 * torch.randn(nlat, nlon, dtype=torch.float64)
    input_data, gen_data = _ssh_salt_states(ocean_mask, input_ssh, input_ssh + rise_m)
    forcing_data = {"sea_surface_fraction": torch.ones(nlat, nlon, device=DEVICE)}
    corrected = corrector(input_data, gen_data, forcing_data, None).corrected

    ocean_cell_area = _ocean_cell_area_m2(ocean_mask)
    torch.testing.assert_close(
        _total_salt_content(corrected, ocean_cell_area, layer_thickness),
        _total_salt_content(input_data, ocean_cell_area, layer_thickness)
        - reference_salinity * rise_m * float(ocean_cell_area.sum()),
        rtol=1e-12,
        atol=0.0,
    )


def test_ocean_salt_content_correction_sea_surface_height_nan_over_land():
    # SSH is NaN over land in the data. That must give the same correction as
    # no height change there, and must not poison the budget even if the
    # global total does not drop the land cells through the ocean mask.
    torch.manual_seed(0)
    ocean_mask = _salt_ocean_mask()
    land = ocean_mask.to(DEVICE) == 0
    nlat, nlon = ocean_mask.shape
    corrector = _ssh_budget_corrector(_salt_dataset_info(ocean_mask, (10.0, 20.0)))
    input_ssh = 0.5 * torch.randn(nlat, nlon, dtype=torch.float64)
    gen_ssh = input_ssh + 1e-3 * torch.randn(nlat, nlon, dtype=torch.float64)
    input_data, gen_data = _ssh_salt_states(ocean_mask, input_ssh, gen_ssh)
    sea_surface_fraction = torch.rand(nlat, nlon, dtype=torch.float64, device=DEVICE)
    forcing_data = {"sea_surface_fraction": sea_surface_fraction.where(~land, 0.0)}

    def corrected_so_0(input_ssh_land: float, gen_ssh_land: float) -> torch.Tensor:
        input_ = dict(input_data, SSH=input_data["SSH"].where(~land, input_ssh_land))
        gen = dict(gen_data, SSH=gen_data["SSH"].where(~land, gen_ssh_land))
        return corrector(input_, gen, forcing_data, None).corrected["so_0"]

    nan_over_land = corrected_so_0(float("nan"), float("nan"))
    assert torch.isfinite(nan_over_land[~land]).all()
    torch.testing.assert_close(nan_over_land, corrected_so_0(0.0, 0.0), equal_nan=True)

    def unmasked_total(data: torch.Tensor) -> torch.Tensor:
        return data.sum(dim=(-2, -1), keepdim=True)

    budget = SeaSurfaceHeightSaltBudget(
        reference_salinity_psu=35.0,
    )
    expected_change = budget(
        OceanData(input_data),
        OceanData(gen_data),
        OceanData(forcing_data),
        unmasked_total,
        timestep_seconds=432000.0,
        dtype=torch.float64,
    )
    assert torch.isfinite(expected_change).all()


def test_ocean_salt_content_correction_sea_surface_height_budget_without_ssh():
    torch.manual_seed(0)
    ocean_mask = _salt_ocean_mask()
    corrector = _ssh_budget_corrector(_salt_dataset_info(ocean_mask, (10.0, 20.0)))
    ssh = torch.zeros(ocean_mask.shape, dtype=torch.float64)
    input_data, gen_data = _ssh_salt_states(ocean_mask, ssh, ssh)
    forcing_data = {"sea_surface_fraction": ocean_mask.to(DEVICE)}
    del input_data["SSH"], gen_data["SSH"]
    with pytest.raises(ValueError, match="SSH.*not zos"):
        corrector(input_data, gen_data, forcing_data, None)


def test_ocean_salt_content_budget_config_rejects_unweighted_sea_surface_height():
    with pytest.raises(ValueError, match="weight_by_sea_surface_fraction"):
        OceanSaltContentBudgetConfig(
            method="scaled_salinity",
            budget_config=SeaSurfaceHeightSaltBudgetConfig(),
            weight_by_sea_surface_fraction=False,
        )


def test_ocean_salt_content_correction_sea_surface_height_budget_float64():
    # At realistic magnitudes (column salt ~1e5 psu m, a budget of ~3e-2 psu m
    # per unit area, i.e. ~1 mm of sea level) the SSH budget is met closely
    # from a float32 state with use_float64, as the ice volume budget is, and
    # the corrected salinity keeps the state's float32 dtype.
    torch.manual_seed(0)
    nlat, nlon = 32, 64
    ocean_mask = torch.ones(nlat, nlon)
    layer_thickness = (1000.0, 3000.0)
    dataset_info = _salt_dataset_info(ocean_mask, layer_thickness)
    reference_salinity = 35.0
    rise_m = 3e-2 / reference_salinity
    sea_surface_fraction = torch.ones(nlat, nlon, device=DEVICE)
    sea_surface_fraction[: nlat // 4] = 0.5  # partly-land cells
    input_ssh = 0.5 * torch.randn(nlat, nlon, device=DEVICE)
    input_so = 35.0 + torch.rand(2, nlat, nlon, device=DEVICE)
    gen_so = input_so + 1e-3 + 1e-4 * torch.randn(2, nlat, nlon, device=DEVICE)
    input_data = {"so_0": input_so[0], "so_1": input_so[1], "SSH": input_ssh}
    gen_data = {"so_0": gen_so[0], "so_1": gen_so[1], "SSH": input_ssh + rise_m}
    forcing_data = {"sea_surface_fraction": sea_surface_fraction}
    sea_surface_area = _ocean_cell_area_m2(ocean_mask) * sea_surface_fraction.double()
    expected_change = -reference_salinity * float(
        ((gen_data["SSH"].double() - input_ssh.double()) * sea_surface_area).sum()
    )

    def miss(use_float64: bool) -> float:
        config = OceanCorrectorConfig(
            ocean_salt_content_correction=OceanSaltContentBudgetConfig(
                method="scaled_salinity",
                use_float64=use_float64,
                budget_config=SeaSurfaceHeightSaltBudgetConfig(
                    reference_salinity_psu=reference_salinity,
                ),
            )
        )
        corrected = config.get_corrector(dataset_info)(
            input_data, gen_data, forcing_data, None
        ).corrected
        assert corrected["so_0"].dtype == torch.float32
        return float(
            _total_salt_content(corrected, sea_surface_area, layer_thickness)
            - _total_salt_content(input_data, sea_surface_area, layer_thickness)
            - expected_change
        )

    assert abs(miss(use_float64=True)) < 0.05 * abs(expected_change)
    assert abs(miss(use_float64=True)) < abs(miss(use_float64=False))


@pytest.mark.parametrize(
    "budget_config, expected",
    [
        pytest.param(None, None, id="none"),
        pytest.param(
            {"type": "sea_surface_height"},
            SeaSurfaceHeightSaltBudgetConfig(),
            id="sea_surface_height",
        ),
        pytest.param(
            {"type": "sea_surface_height", "reference_salinity_psu": 34.0},
            SeaSurfaceHeightSaltBudgetConfig(reference_salinity_psu=34.0),
            id="sea_surface_height_reference_salinity",
        ),
    ],
)
def test_ocean_salt_content_budget_config_from_state(budget_config, expected):
    state = {"method": "scaled_salinity", "budget_config": budget_config}
    config = OceanCorrectorConfig.from_state({"ocean_salt_content_correction": state})
    assert config.ocean_salt_content_correction is not None
    assert config.ocean_salt_content_correction.budget_config == expected


def test_ocean_corrector_config_fields_are_known():
    # Staleness guard: if a new corrector option is added to
    # OceanCorrectorConfig this fails, flagging that the corrector delta/
    # modified-return tests need to exercise it.
    expected = {
        "force_positive_names",
        "sea_ice_fraction_correction",
        "surface_energy_flux_correction",
        "ocean_heat_content_correction",
        "ocean_salt_content_correction",
        "keep_gradient_through_clamps",
        "zos_global_mean_correction",
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
        ocean_salt_content_correction=OceanSaltContentBudgetConfig(
            method="scaled_salinity",
            budget_config=SeaSurfaceHeightSaltBudgetConfig(),
            constant_unaccounted_salting=1e-9,
        ),
        zos_global_mean_correction=ZosGlobalMeanCorrectionConfig(
            reference_global_mean=0.01
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
        "so_0": randoms((n_members, nlat, nlon)) + 35.0,
        "so_1": randoms((n_members, nlat, nlon)) + 35.0,
        "SSH": randoms((n_members, nlat, nlon)),
        "land_fraction": torch.zeros(n_members, nlat, nlon),
    }
    # members differ in every generated field, as they would under different
    # noise draws
    gen_data = {
        "thetao_0": randoms((n_members, nlat, nlon)) + 2.0,
        "thetao_1": randoms((n_members, nlat, nlon)) + 2.0,
        "sst": randoms((n_members, nlat, nlon)) + 275.0,
        "so_0": randoms((n_members, nlat, nlon)) + 35.0,
        "so_1": randoms((n_members, nlat, nlon)) + 35.0,
        "SSH": randoms((n_members, nlat, nlon)),
        # spans the clamp range at both ends so the sea-ice rebalance engages
        "sea_ice_fraction": randoms((n_members, nlat, nlon)) * 0.8 + 0.5,
        "sea_ice_thickness": randoms((n_members, nlat, nlon)),
        "hfds": randoms((n_members, nlat, nlon)),
        "zos": randoms((n_members, nlat, nlon)),
    }
    forcing_data = {
        "hfgeou": randoms((n_members, nlat, nlon)),
        "sea_surface_fraction": mask[:, :, 0].expand(n_members, nlat, nlon),
    }

    folded = corrector(input_data, gen_data, forcing_data, None).corrected
    assert set(folded) >= {"so_0", "sea_ice_fraction", "thetao_0", "sst", "zos"}

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


def _zos_setup(reference_global_mean: float):
    nsamples, nlat, nlon = 3, 4, 5
    mask = torch.ones(nlat, nlon, device=DEVICE)
    mask[0, :2] = 0.0
    mask[3, 4] = 0.0
    # fractional at masked-in coastal cells, zero off the mask
    sea_surface_fraction = mask.clone()
    sea_surface_fraction[0, 2] = 0.3
    sea_surface_fraction[1, 0] = 0.6
    sea_surface_fraction[2, 4] = 0.1
    area = torch.tensor([0.5, 1.0, 1.5, 0.7], device=DEVICE).unsqueeze(-1)
    area = area.expand(nlat, nlon)
    ops = LatLonOperations(area, SpatialMaskProvider({"mask_2d": mask}))
    config = OceanCorrectorConfig(
        zos_global_mean_correction=ZosGlobalMeanCorrectionConfig(
            reference_global_mean=reference_global_mean
        )
    )
    corrector = config._build(ops, None, datetime.timedelta(seconds=3600))
    torch.manual_seed(0)
    gen_data = {
        "zos": torch.randn(nsamples, nlat, nlon, device=DEVICE) + 0.5,
        "so_0": torch.randn(nsamples, nlat, nlon, device=DEVICE),
    }
    forcing_data = {
        "sea_surface_fraction": sea_surface_fraction.expand(nsamples, nlat, nlon)
    }
    return corrector, gen_data, forcing_data, area, sea_surface_fraction


@pytest.mark.parametrize("reference_global_mean", [0.0, -0.0123])
def test_zos_global_mean_correction(reference_global_mean):
    corrector, gen_data, forcing_data, area, s = _zos_setup(reference_global_mean)
    result = corrector({}, gen_data, forcing_data, None)
    zos = result.corrected["zos"]
    # <z>_f = sum A s z / sum A s, independent of fme's area_weighted_mean
    w = area * s
    mean_f = (w * zos).sum(dim=(-2, -1)) / w.sum()
    torch.testing.assert_close(
        mean_f, torch.full_like(mean_f, reference_global_mean), atol=1e-6, rtol=0
    )
    # a uniform shift per sample
    shift = zos - gen_data["zos"]
    torch.testing.assert_close(
        shift, shift[:, :1, :1].expand_as(shift), atol=1e-6, rtol=0
    )
    assert set(result.modified_names) == {"zos"}
    torch.testing.assert_close(result.corrected["so_0"], gen_data["so_0"])


def test_zos_global_mean_correction_differs_from_mask_weighting():
    # guards the weighting: with fractional s, the mask-weighted mean of the
    # corrected zos is not the reference, so <z>_mask would be the wrong choice
    corrector, gen_data, forcing_data, area, s = _zos_setup(0.0)
    zos = corrector({}, gen_data, forcing_data, None).corrected["zos"]
    mask = (s > 0).to(area.dtype)
    mean_mask = (area * mask * zos).sum(dim=(-2, -1)) / (area * mask).sum()
    assert mean_mask.abs().min() > 1e-4


def test_zos_global_mean_correction_absent_zos():
    corrector, gen_data, forcing_data, _, _ = _zos_setup(0.0)
    del gen_data["zos"]
    result = corrector({}, gen_data, forcing_data, None)
    assert set(result.modified_names) == set()
    assert set(result.corrected) == {"so_0"}
    torch.testing.assert_close(result.corrected["so_0"], gen_data["so_0"])


def test_zos_global_mean_correction_config_round_trip():
    config = OceanCorrectorConfig(
        zos_global_mean_correction=ZosGlobalMeanCorrectionConfig(
            reference_global_mean=-0.0123
        )
    )
    # None-valued keys dropped: remove_deprecated_keys does not accept an
    # explicit ocean_heat_content_correction=None
    state = {k: v for k, v in dataclasses.asdict(config).items() if v is not None}
    assert state["zos_global_mean_correction"] == {"reference_global_mean": -0.0123}
    assert OceanCorrectorConfig.from_state(state) == config
    default = OceanCorrectorConfig.from_state({"zos_global_mean_correction": {}})
    assert default.zos_global_mean_correction == ZosGlobalMeanCorrectionConfig(
        reference_global_mean=0.0
    )
    assert OceanCorrectorConfig.from_state({}).zos_global_mean_correction is None
