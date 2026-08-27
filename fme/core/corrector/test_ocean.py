import dataclasses
import datetime

import pytest
import torch

from fme import get_device
from fme.core.coordinates import DepthCoordinate
from fme.core.corrector.ocean import (
    OceanCorrectorConfig,
    OceanHeatContentBudgetConfig,
    OceanHeatContentWeightsConfig,
    SeaIceFractionConfig,
    SurfaceEnergyFluxCorrectionConfig,
    ZosGlobalMeanCorrectionConfig,
    _compute_ocean_net_surface_energy_flux,
    _mixed_layer_depth,
    _weighted_temperature_correction,
)
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


def _make_prescribed_hfds_total_area_setup():
    """Setup for the prescribed-hfds_total_area tests: masked ops, depth
    coordinate, and input/gen/forcing dicts for a corrector with both the
    surface energy flux and ocean heat content corrections active."""
    timestep = datetime.timedelta(seconds=5 * 24 * 3600)
    nsamples, nlat, nlon, nlevels = 4, 3, 3, 2
    mask = torch.ones(nlat, nlon, nlevels, device=DEVICE)
    mask[0, 0, :] = 0.0
    masks = {
        "mask_0": mask[:, :, 0],
        "mask_1": mask[:, :, 1],
        "mask_2d": mask[:, :, 0],
    }
    ops = LatLonOperations(torch.ones(nlat, nlon), SpatialMaskProvider(masks))
    idepth = torch.tensor([2.5, 10, 20], device=DEVICE)
    depth_coordinate = DepthCoordinate(idepth, mask)
    sea_surface_fraction = mask[:, :, 0]
    land_fraction = 1 - sea_surface_fraction
    input_data = {
        "thetao_0": torch.ones(nsamples, nlat, nlon, device=DEVICE),
        "thetao_1": torch.ones(nsamples, nlat, nlon, device=DEVICE),
        "sst": torch.ones(nsamples, nlat, nlon, device=DEVICE) + 273.15,
        "land_fraction": land_fraction,
        "sea_ice_fraction": torch.zeros(nsamples, nlat, nlon, device=DEVICE),
    }
    gen_data = {
        "thetao_0": torch.ones(nsamples, nlat, nlon, device=DEVICE) * 2,
        "thetao_1": torch.ones(nsamples, nlat, nlon, device=DEVICE) * 2,
        "sst": torch.ones(nsamples, nlat, nlon, device=DEVICE) * 2 + 273.15,
        "hfds_total_area": torch.full((nsamples, nlat, nlon), 5.0, device=DEVICE),
    }
    forcing_data = {
        "land_fraction": land_fraction,
        "sea_surface_fraction": sea_surface_fraction,
        "hfgeou": torch.ones(nsamples, nlat, nlon, device=DEVICE),
        **_make_atmos_forcing_data((nsamples, nlat, nlon)),
    }
    return timestep, ops, depth_coordinate, input_data, gen_data, forcing_data


def _prescribed_hfds_total_area_target(forcing_data):
    """A target hfds_total_area with NaN over land, as in the zarr stores."""
    sea_surface_fraction = forcing_data["sea_surface_fraction"]
    target = torch.full_like(forcing_data["hfgeou"], 7.0) * sea_surface_fraction
    return torch.where(
        sea_surface_fraction > 0,
        target,
        torch.full_like(target, float("nan")),
    )


def test_prescribed_hfds_total_area_forcing_overrides_correction():
    (
        timestep,
        ops,
        depth_coordinate,
        input_data,
        gen_data,
        forcing_data,
    ) = _make_prescribed_hfds_total_area_setup()
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed"
        ),
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="scaled_temperature"
        ),
    )
    corrector = config._build(ops, depth_coordinate, timestep)
    target = _prescribed_hfds_total_area_target(forcing_data)
    forcing_data = {**forcing_data, "hfds_total_area": target}

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected

    # the corrector emits the forcing value verbatim (NaN over land included)
    torch.testing.assert_close(corrected["hfds_total_area"], target, equal_nan=True)
    # the OHC correction scales temperature against the prescribed flux; the
    # masked area mean keeps the land NaN out of the budget
    flux = target + forcing_data["hfgeou"] * forcing_data["sea_surface_fraction"]
    flux_mean = ops.area_weighted_mean(flux, keepdim=True, name="ocean_heat_content")
    input_ohc = ops.area_weighted_mean(
        OceanData(input_data, depth_coordinate).ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    gen_ohc = ops.area_weighted_mean(
        OceanData(gen_data, depth_coordinate).ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    ratio = (input_ohc + flux_mean * timestep.total_seconds()) / gen_ohc
    assert torch.isfinite(ratio).all()
    for name in ["thetao_0", "thetao_1"]:
        torch.testing.assert_close(corrected[name], gen_data[name] * ratio)
    torch.testing.assert_close(
        corrected["sst"], (gen_data["sst"] - 273.15) * ratio + 273.15
    )


def test_prescribed_hfds_total_area_absent_keeps_existing_correction():
    (
        timestep,
        ops,
        depth_coordinate,
        input_data,
        gen_data,
        forcing_data,
    ) = _make_prescribed_hfds_total_area_setup()
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed"
        ),
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="scaled_temperature"
        ),
    )
    corrector = config._build(ops, depth_coordinate, timestep)

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected

    ocean_fraction = 1 - input_data["land_fraction"] - input_data["sea_ice_fraction"]
    net_flux = (
        _compute_ocean_net_surface_energy_flux(forcing_data, input_data["sst"])
        * forcing_data["sea_surface_fraction"]
    )
    expected = net_flux * ocean_fraction + gen_data["hfds_total_area"] * (
        1 - ocean_fraction
    )
    torch.testing.assert_close(corrected["hfds_total_area"], expected)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            OceanCorrectorConfig(
                surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
                    method="prescribed"
                )
            ),
            id="missing_ohc_correction",
        ),
        pytest.param(
            OceanCorrectorConfig(
                ocean_heat_content_correction=OceanHeatContentBudgetConfig(
                    method="scaled_temperature"
                )
            ),
            id="missing_surface_energy_flux_correction",
        ),
    ],
)
def test_prescribed_hfds_total_area_requires_both_corrections(config):
    (
        timestep,
        ops,
        depth_coordinate,
        input_data,
        gen_data,
        forcing_data,
    ) = _make_prescribed_hfds_total_area_setup()
    corrector = config._build(ops, depth_coordinate, timestep)
    target = _prescribed_hfds_total_area_target(forcing_data)
    forcing_data = {**forcing_data, "hfds_total_area": target}
    with pytest.raises(ValueError, match="hfds_total_area is prescribed"):
        corrector(input_data, gen_data, forcing_data, None)


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


# ---------------------------------------------------------------------------
# weighted_temperature OHC correction


@dataclasses.dataclass
class _DzeffDepthCoordinate(DepthCoordinate):
    """DepthCoordinate whose ``dz`` is a substituted static (a dzeff stand-in)."""

    dzeff: torch.Tensor | None = None

    @property
    def dz(self) -> torch.Tensor:
        assert self.dzeff is not None
        return self.dzeff


_WT_TIMESTEP = datetime.timedelta(seconds=5 * 24 * 3600)


def _weighted_setup(dzeff: bool = False, finite_input: bool = False):
    """Grid with a land column, mask_k = 1 below deptho, partial bottom cells,
    NaN below the bottom in the input state, and a network-like finite gen."""
    dtype = torch.float64
    torch.manual_seed(0)
    nsamples, nlat, nlon = 2, 3, 4
    idepth = torch.tensor([0.0, 10.0, 30.0, 60.0, 100.0], dtype=dtype, device=DEVICE)
    nz = len(idepth) - 1
    deptho = torch.tensor(
        [
            [100.0, 45.0, 20.0, 75.0],
            [100.0, float("nan"), 60.0, 12.0],
            [35.0, 100.0, 90.0, 100.0],
        ],
        dtype=dtype,
        device=DEVICE,
    )
    land = torch.isnan(deptho)
    mask = (~land).to(dtype).unsqueeze(-1).expand(nlat, nlon, nz).clone()
    masks = {f"mask_{k}": mask[..., k] for k in range(nz)}
    masks["mask_2d"] = mask[..., 0]
    area = torch.linspace(0.5, 1.5, nlat, dtype=dtype, device=DEVICE)
    ops = LatLonOperations(
        area.unsqueeze(-1).expand(nlat, nlon), SpatialMaskProvider(masks)
    )
    coord = DepthCoordinate(idepth, mask, deptho)
    if dzeff:
        factor = 0.8 + 0.4 * torch.rand(nlat, nlon, nz, dtype=dtype, device=DEVICE)
        dz = coord.dz * factor
        dz = torch.where(dz > 0, dz, torch.full_like(dz, float("nan")))
        coord = _DzeffDepthCoordinate(idepth, mask, deptho, dzeff=dz)
    zc = 0.5 * (idepth[:-1] + idepth[1:])
    h = 5.0 + 50.0 * torch.rand(nsamples, nlat, nlon, 1, dtype=dtype, device=DEVICE)
    t0 = 15.0 + 5.0 * torch.rand(nsamples, nlat, nlon, 1, dtype=dtype, device=DEVICE)
    thetao_in = t0 - 0.1 * torch.clamp(zc - h, min=0.0)
    so_in = 35.0 + 0.001 * zc.expand_as(thetao_in)
    thetao_gen = thetao_in + 0.3 * torch.randn_like(thetao_in) + 0.5
    if not finite_input:
        below = (idepth[:-1] >= torch.nan_to_num(deptho, nan=0.0).unsqueeze(-1)) | (
            land.unsqueeze(-1)
        )
        thetao_in = thetao_in.masked_fill(below, float("nan"))
        so_in = so_in.masked_fill(below, float("nan"))
    sst_in = thetao_in[..., 0] + 273.15
    sst_gen = thetao_gen[..., 0] + 0.2 + 273.15
    input_data = {f"thetao_{k}": thetao_in[..., k] for k in range(nz)}
    input_data.update({f"so_{k}": so_in[..., k] for k in range(nz)})
    input_data["sst"] = sst_in
    gen_data = {f"thetao_{k}": thetao_gen[..., k] for k in range(nz)}
    gen_data["sst"] = sst_gen
    ssf = mask[..., 0].expand(nsamples, nlat, nlon)
    gen_data["hfds_total_area"] = 50.0 * ssf
    forcing_data = {
        "hfgeou": torch.full_like(ssf, 0.1),
        "sea_surface_fraction": ssf,
    }
    return ops, coord, input_data, gen_data, forcing_data


def _global_ohc(data, coord, ops):
    return ops.area_weighted_mean(
        OceanData(data, coord).ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )


def _expected_change(ops, gen_data, forcing_data, unaccounted):
    flux = (
        gen_data["hfds_total_area"]
        + forcing_data["hfgeou"] * forcing_data["sea_surface_fraction"]
    )
    mean = ops.area_weighted_mean(flux, keepdim=True, name="ocean_heat_content")
    return (mean + unaccounted) * _WT_TIMESTEP.total_seconds()


def _weighted_config(weights_type, **kwargs):
    return OceanCorrectorConfig(
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="weighted_temperature",
            constant_unaccounted_heating=0.1,
            weights=OceanHeatContentWeightsConfig(type=weights_type),
            **kwargs,
        )
    )


@pytest.mark.parametrize("dzeff", [False, True], ids=["default_dz", "dzeff"])
@pytest.mark.parametrize("weights_type", ["mld", "theta"])
def test_weighted_temperature_closure(weights_type, dzeff):
    ops, coord, input_data, gen_data, forcing_data = _weighted_setup(dzeff=dzeff)
    corrector = _weighted_config(weights_type)._build(ops, coord, _WT_TIMESTEP)
    result = corrector(input_data, gen_data, forcing_data, None)
    assert set(result.modified_names) == {f"thetao_{k}" for k in range(4)} | {"sst"}
    corrected = {**gen_data, **result.corrected}
    ohc_in = _global_ohc(input_data, coord, ops)
    r = (
        _global_ohc(corrected, coord, ops)
        - ohc_in
        - _expected_change(ops, gen_data, forcing_data, 0.1)
    )
    assert torch.isfinite(r).all()
    torch.testing.assert_close(
        r, torch.zeros_like(r), atol=1e-12 * ohc_in.abs().max().item(), rtol=0
    )
    # the gen budget was not already closed, so the correction did something
    dE = _expected_change(ops, gen_data, forcing_data, 0.1) + ohc_in
    dE = dE - _global_ohc(gen_data, coord, ops)
    assert dE.abs().min() > 1e-3 * ohc_in.abs().max()


@pytest.mark.parametrize("dzeff", [False, True], ids=["default_dz", "dzeff"])
def test_weighted_temperature_mld_shape(dzeff):
    ops, coord, input_data, gen_data, forcing_data = _weighted_setup(dzeff=dzeff)
    input = OceanData(input_data, coord)
    gen = OceanData(gen_data, coord)
    weights = OceanHeatContentWeightsConfig(type="mld")
    out, diag = _weighted_temperature_correction(
        input,
        gen,
        _expected_change(ops, gen_data, forcing_data, 0.1),
        ops.area_weighted_mean,
        coord,
        _WT_TIMESTEP.total_seconds(),
        weights,
        detach_weights=True,
    )
    # independent w: clamp((min(MLD, deptho) - z_top) / dz, 0, 1) on the support
    mld = _mixed_layer_depth(
        input.sea_water_potential_temperature,
        input.sea_water_salinity,
        coord.idepth,
        coord.mask,
        coord.deptho,
        weights.delta_rho_threshold,
        weights.mld_ref_layer,
    )
    dz = coord.dz
    support = (coord.mask > 0) & torch.isfinite(dz) & (dz > 0)
    m = torch.minimum(mld, coord.deptho).unsqueeze(-1)
    w = torch.where(
        support,
        torch.clamp((m - coord.idepth[:-1]) / torch.where(support, dz, 1.0), 0, 1),
        0.0,
    )
    thetao_gen = gen.sea_water_potential_temperature
    dT = OceanData(out, coord).sea_water_potential_temperature - thetao_gen
    c = diag.c.unsqueeze(-1)
    torch.testing.assert_close(dT, c * w)
    assert (dT.abs() <= c.abs() * (1 + 1e-12)).all()
    zero_w = w == 0
    assert zero_w.any() and (~zero_w).any()
    assert ((w > 0) & (w < 1)).any()
    assert (dT[zero_w] == 0).all()
    torch.testing.assert_close(out["sst"] - gen_data["sst"], diag.c * w[..., 0])
    # diagnostics
    torch.testing.assert_close(
        diag.max_abs_dT, dT.abs().amax(dim=(-3, -2, -1)).reshape(diag.c.shape)
    )
    torch.testing.assert_close(
        diag.closure_residual,
        torch.zeros_like(diag.closure_residual),
        atol=1e-12 * _global_ohc(input_data, coord, ops).abs().max().item(),
        rtol=0,
    )
    dE = diag.c * diag.denominator
    torch.testing.assert_close(diag.raw_adv, -dE / _WT_TIMESTEP.total_seconds())
    assert diag.mean_mld is not None and torch.isfinite(diag.mean_mld).all()


def test_weighted_temperature_theta_matches_scaled_temperature():
    ops, coord, input_data, gen_data, forcing_data = _weighted_setup()
    weighted = _weighted_config("theta")._build(ops, coord, _WT_TIMESTEP)
    scaled = OceanCorrectorConfig(
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="scaled_temperature", constant_unaccounted_heating=0.1
        )
    )._build(ops, coord, _WT_TIMESTEP)
    out_w = weighted(input_data, gen_data, forcing_data, None).corrected
    out_s = scaled(input_data, gen_data, forcing_data, None).corrected
    assert set(out_w) == set(out_s)
    support = coord.mask > 0
    for k in range(4):
        sk = support[..., k].expand_as(out_w[f"thetao_{k}"])
        torch.testing.assert_close(out_w[f"thetao_{k}"][sk], out_s[f"thetao_{k}"][sk])
    s0 = support[..., 0].expand_as(out_w["sst"])
    torch.testing.assert_close(out_w["sst"][s0], out_s["sst"][s0])


def test_weighted_temperature_zero_weights_raise():
    ops, coord, input_data, gen_data, forcing_data = _weighted_setup()
    for k in range(4):
        gen_data[f"thetao_{k}"] = torch.zeros_like(gen_data[f"thetao_{k}"])
    corrector = _weighted_config("theta")._build(ops, coord, _WT_TIMESTEP)
    with pytest.raises(ValueError, match="denominator"):
        corrector(input_data, gen_data, forcing_data, None)


def test_weighted_temperature_requires_depth_geometry():
    ops, _, input_data, gen_data, forcing_data = _weighted_setup()
    corrector = _weighted_config("mld")._build(ops, _MockDepth(), _WT_TIMESTEP)
    with pytest.raises(ValueError, match="requires a depth coordinate"):
        corrector(input_data, gen_data, forcing_data, None)


def _gradients(weights_type, detach_weights):
    ops, coord, input_data, gen_data, forcing_data = _weighted_setup(finite_input=True)
    gen_data = {k: v.clone().requires_grad_() for k, v in gen_data.items()}
    so = {
        k: v.clone().requires_grad_()
        for k, v in input_data.items()
        if k.startswith("so_")
    }
    input_data = {**input_data, **so}
    corrector = _weighted_config(weights_type, detach_weights=detach_weights)._build(
        ops, coord, _WT_TIMESTEP
    )
    out = corrector(input_data, gen_data, forcing_data, None).corrected
    loss = sum(
        (v * torch.linspace(1.0, 2.0, v.shape[-1], dtype=v.dtype)).sum()
        for k, v in out.items()
    )
    loss.backward()
    return gen_data, so


@pytest.mark.parametrize("weights_type", ["mld", "theta"])
def test_weighted_temperature_detach_weights_gradient(weights_type):
    gen_det, so_det = _gradients(weights_type, detach_weights=True)
    gen_live, so_live = _gradients(weights_type, detach_weights=False)
    for k in range(4):
        g = gen_det[f"thetao_{k}"].grad
        assert g is not None and torch.isfinite(g).all()
    # loss reaches thetao_gen through c: the gradient is not that of the
    # identity map (which would be the loss weights alone)
    g0 = gen_det["thetao_0"].grad
    assert not torch.allclose(
        g0, torch.linspace(1.0, 2.0, g0.shape[-1], dtype=g0.dtype).expand_as(g0)
    )
    if weights_type == "mld":
        # so_in enters only through w
        assert all(v.grad is None for v in so_det.values())
        assert any(v.grad is not None for v in so_live.values())
        assert all(
            torch.isfinite(v.grad).all() for v in so_live.values() if v.grad is not None
        )
    else:
        assert not torch.allclose(gen_det["thetao_0"].grad, gen_live["thetao_0"].grad)


def test_weighted_temperature_mld_live_weights_nan_input_gradient_finite():
    ops, coord, input_data, gen_data, forcing_data = _weighted_setup()
    input_data = {
        k: v.clone().requires_grad_() if k.startswith(("thetao_", "so_")) else v
        for k, v in input_data.items()
    }
    gen_data = {k: v.clone().requires_grad_() for k, v in gen_data.items()}
    corrector = _weighted_config("mld", detach_weights=False)._build(
        ops, coord, _WT_TIMESTEP
    )
    out = corrector(input_data, gen_data, forcing_data, None).corrected
    sum(v.nansum() for v in out.values()).backward()
    for v in gen_data.values():
        assert v.grad is not None and torch.isfinite(v.grad).all()
    so_grads = [input_data[f"so_{k}"].grad for k in range(4)]
    assert any(g is not None for g in so_grads)
    for k, v in input_data.items():
        if k.startswith(("thetao_", "so_")) and v.grad is not None:
            finite = torch.isfinite(v.detach())
            assert torch.isfinite(v.grad[finite]).all(), k


def test_weighted_temperature_config_round_trip():
    config = _weighted_config("mld", detach_weights=False)
    state = {k: v for k, v in dataclasses.asdict(config).items() if v is not None}
    assert OceanCorrectorConfig.from_state(state) == config
    legacy = OceanCorrectorConfig.from_state(
        {"ocean_heat_content_correction": {"method": "scaled_temperature"}}
    )
    ohc = legacy.ocean_heat_content_correction
    assert ohc is not None
    assert ohc.method == "scaled_temperature"
    assert ohc.weights == OceanHeatContentWeightsConfig()
    assert ohc.detach_weights is True
