import dataclasses
import datetime
from unittest.mock import PropertyMock, patch

import pytest
import torch

from fme import get_device
from fme.core.constants import EARTH_RADIUS
from fme.core.coordinates import DepthCoordinate, LatLonCoordinates
from fme.core.corrector.ocean import (
    IceVolumeSaltBudgetConfig,
    OceanCorrectorConfig,
    OceanHeatContentBudgetConfig,
    OceanSaltContentBudgetConfig,
    SeaIceFractionConfig,
    SeaSurfaceHeightSaltBudget,
    SeaSurfaceHeightSaltBudgetConfig,
    SurfaceEnergyFluxCorrectionConfig,
    WaterFluxRegimesConfig,
    WaterFluxSaltBudgetConfig,
    _compute_ocean_net_surface_energy_flux,
)
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.gridded_ops import LatLonOperations
from fme.core.ocean_data import OceanData
from fme.core.registry.corrector import CorrectorSelector
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


def _salt_coordinates(nlat: int, nlon: int) -> LatLonCoordinates:
    return LatLonCoordinates(
        lat=torch.linspace(-80.0, 80.0, nlat), lon=torch.arange(nlon) * 360.0 / nlon
    )


def _salt_dataset_info(
    ocean_mask: torch.Tensor,
    layer_thickness: tuple[float, float],
    sea_ice_volume_mask: torch.Tensor | None = None,
) -> DatasetInfo:
    """Lat-lon grid with non-uniform cell areas, a two-layer depth coordinate,
    and an optional sea_ice_volume mask."""
    nlat, nlon = ocean_mask.shape
    masks = {"mask_0": ocean_mask, "mask_1": ocean_mask, "mask_2d": ocean_mask}
    if sea_ice_volume_mask is not None:
        masks["mask_sea_ice_volume"] = sea_ice_volume_mask
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


def _salt_ocean_and_ice_masks() -> tuple[torch.Tensor, torch.Tensor]:
    ocean_mask = torch.ones(4, 8)
    ocean_mask[1, 2] = 0.0  # a land cell
    ice_mask = ocean_mask.clone()  # as in the data, no ice data over land
    ice_mask[1:3] = 0.0  # or in the tropics
    return ocean_mask, ice_mask


def _ice_volume_salt_config(slope_psu: float, **kwargs) -> OceanSaltContentBudgetConfig:
    """The ice volume budget on the unweighted salt content, or no budget for a
    zero slope."""
    kwargs.setdefault("weight_by_sea_surface_fraction", False)
    return OceanSaltContentBudgetConfig(
        method="scaled_salinity",
        budget_config=(
            None if slope_psu == 0.0 else IceVolumeSaltBudgetConfig(slope_psu)
        ),
        **kwargs,
    )


def test_ocean_salt_content_correction():
    torch.manual_seed(0)
    ocean_mask, ice_mask = _salt_ocean_and_ice_masks()
    nlat, nlon = ocean_mask.shape
    layer_thickness = (10.0, 20.0)
    dataset_info = _salt_dataset_info(ocean_mask, layer_thickness, ice_mask)
    ocean_cell_area = _ocean_cell_area_m2(ocean_mask)
    slope, constant = 40.0, 3e-9
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=_ice_volume_salt_config(
            slope, constant_unaccounted_salting=constant
        )
    )
    corrector = config.get_corrector(dataset_info)

    def salinity(value):
        so = value + torch.rand(2, nlat, nlon, dtype=torch.float64, device=DEVICE)
        return so.where(ocean_mask.to(DEVICE) > 0, float("nan"))

    input_ice = torch.rand(nlat, nlon, dtype=torch.float64) * 1e10 * ice_mask
    gen_ice = input_ice + torch.rand(nlat, nlon, dtype=torch.float64) * 1e9 * ice_mask
    input_so, gen_so = salinity(34.0), salinity(35.0)
    input_data = {
        "so_0": input_so[0],
        "so_1": input_so[1],
        "sea_ice_volume": input_ice.to(DEVICE),
    }
    gen_data = {
        "so_0": gen_so[0],
        "so_1": gen_so[1],
        "sea_ice_volume": gen_ice.to(DEVICE),
    }
    result = corrector(input_data, gen_data, {}, None)

    # the salt correction writes every salinity level; sea_ice_volume is read
    # but not written
    assert set(result.modified_names) == {"so_0", "so_1"}
    for name, delta in result.diagnostics.delta.items():
        torch.testing.assert_close(
            delta, result.corrected[name] - gen_data[name], equal_nan=True
        )

    # the ice term is the plain sum of the ice volume change, and the constant
    # rate applies over the ocean area
    expected_change = slope * float(
        (gen_ice - input_ice).sum()
    ) + constant * dataset_info.timestep.total_seconds() * float(ocean_cell_area.sum())
    torch.testing.assert_close(
        _total_salt_content(result.corrected, ocean_cell_area, layer_thickness),
        _total_salt_content(input_data, ocean_cell_area, layer_thickness)
        + expected_change,
        rtol=1e-12,
        atol=0.0,
    )
    # by one ratio applied to every level
    ratio = result.corrected["so_0"] / gen_data["so_0"]
    torch.testing.assert_close(
        result.corrected["so_1"], gen_data["so_1"] * ratio, equal_nan=True
    )


def test_ocean_salt_content_correction_ignores_ice_outside_mask():
    # The stepper's output masking leaves sea_ice_volume NaN outside its mask
    # (or input masking fills it), but the prediction there is unconstrained, so
    # neither must count as an ice change.
    torch.manual_seed(0)
    ocean_mask, ice_mask = _salt_ocean_and_ice_masks()
    # a fractional mask value, which the output masking rounds to outside
    ice_mask[1, 0] = 0.4
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=_ice_volume_salt_config(40.0)
    )
    corrector = config.get_corrector(
        _salt_dataset_info(ocean_mask, (10.0, 20.0), ice_mask)
    )
    valid = torch.round(ice_mask.to(DEVICE)) != 0
    input_data = {
        "so_0": 34.0 + torch.rand(ocean_mask.shape, device=DEVICE),
        "so_1": 34.0 + torch.rand(ocean_mask.shape, device=DEVICE),
        "sea_ice_volume": (torch.rand(ocean_mask.shape, device=DEVICE) * 1e10).where(
            valid, float("nan")
        ),
    }
    gen_ice = torch.rand(ocean_mask.shape, device=DEVICE) * 1e10 * valid

    def corrected_so_0(sea_ice_volume):
        gen_data = dict(input_data, sea_ice_volume=sea_ice_volume)
        return corrector(input_data, gen_data, {}, None).corrected["so_0"]

    outside_mask_ice = torch.rand(ocean_mask.shape, device=DEVICE) * 1e12 * ~valid
    torch.testing.assert_close(
        corrected_so_0(gen_ice + outside_mask_ice), corrected_so_0(gen_ice)
    )
    # while the same ice moved inside the mask does count
    assert not torch.allclose(
        corrected_so_0(gen_ice + outside_mask_ice.roll(2, dims=0)),
        corrected_so_0(gen_ice),
    )


@pytest.mark.parametrize(
    "spatial_mask_provider",
    [
        pytest.param(None, id="no_mask_provider"),
        pytest.param(SpatialMaskProvider({"mask_0": torch.ones(4, 8)}), id="no_mask"),
    ],
)
def test_ocean_salt_content_correction_counts_all_ice_without_mask(
    spatial_mask_provider,
):
    # With no mask for sea_ice_volume the stepper keeps the prediction in every
    # cell, so all of the ice volume change counts toward the budget.
    torch.manual_seed(0)
    nlat, nlon = 4, 8
    layer_thickness = (10.0, 20.0)
    dataset_info = DatasetInfo(
        horizontal_coordinates=_salt_coordinates(nlat, nlon),
        vertical_coordinate=DepthCoordinate(
            torch.tensor([0.0, 10.0, 30.0], device=DEVICE),
            torch.ones(nlat, nlon, 2, device=DEVICE),
        ),
        spatial_mask_provider=spatial_mask_provider,
        timestep=datetime.timedelta(seconds=5 * 24 * 3600),
    )
    slope = 40.0
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=_ice_volume_salt_config(slope)
    )
    corrector = config.get_corrector(dataset_info)
    input_data = {
        "so_0": 34.0 + torch.rand(nlat, nlon, dtype=torch.float64, device=DEVICE),
        "so_1": 34.0 + torch.rand(nlat, nlon, dtype=torch.float64, device=DEVICE),
        "sea_ice_volume": torch.zeros(nlat, nlon, dtype=torch.float64, device=DEVICE),
    }
    gen_ice = torch.rand(nlat, nlon, dtype=torch.float64, device=DEVICE) * 1e10
    gen_data = {
        "so_0": input_data["so_0"] + 1.0,
        "so_1": input_data["so_1"] + 1.0,
        "sea_ice_volume": gen_ice,
    }
    corrected = corrector(input_data, gen_data, {}, None).corrected

    ocean_cell_area = _ocean_cell_area_m2(torch.ones(nlat, nlon))
    torch.testing.assert_close(
        _total_salt_content(corrected, ocean_cell_area, layer_thickness),
        _total_salt_content(input_data, ocean_cell_area, layer_thickness)
        + slope * float(gen_ice.sum()),
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize("slope", [0.0, 40.0])
def test_ocean_salt_content_correction_without_sea_ice_volume(slope):
    # Without sea_ice_volume a zero slope holds salt content fixed, while a
    # nonzero slope raises rather than silently dropping the ice term.
    torch.manual_seed(0)
    ocean_mask, _ = _salt_ocean_and_ice_masks()
    layer_thickness = (10.0, 20.0)
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=_ice_volume_salt_config(slope)
    )
    corrector = config.get_corrector(_salt_dataset_info(ocean_mask, layer_thickness))
    input_data = {
        "so_0": 34.0 + torch.rand(ocean_mask.shape, device=DEVICE),
        "so_1": 34.0 + torch.rand(ocean_mask.shape, device=DEVICE),
    }
    gen_data = {name: value + 1.0 for name, value in input_data.items()}
    if slope != 0.0:
        with pytest.raises(ValueError, match="sea_ice_volume is required"):
            corrector(input_data, gen_data, {}, None)
    else:
        corrected = corrector(input_data, gen_data, {}, None).corrected
        ocean_cell_area = _ocean_cell_area_m2(ocean_mask)
        torch.testing.assert_close(
            _total_salt_content(corrected, ocean_cell_area, layer_thickness),
            _total_salt_content(input_data, ocean_cell_area, layer_thickness),
        )


def test_ocean_salt_content_correction_weights_content_by_sea_surface_fraction():
    # Salinity is an ocean-area mean, so with weight_by_sea_surface_fraction the
    # content of a partly-land cell counts in proportion to its ocean part, and
    # the constant term applies over the sea surface area rather than the whole
    # cell area. The sea surface fraction is read from the forcing data.
    torch.manual_seed(0)
    ocean_mask, ice_mask = _salt_ocean_and_ice_masks()
    nlat, nlon = ocean_mask.shape
    layer_thickness = (10.0, 20.0)
    dataset_info = _salt_dataset_info(ocean_mask, layer_thickness, ice_mask)
    ocean_cell_area = _ocean_cell_area_m2(ocean_mask)
    sea_surface_fraction = torch.rand(nlat, nlon, dtype=torch.float64) * ocean_mask
    sea_surface_fraction[0, :] = 1.0  # some wholly-ocean cells too
    slope, constant = 40.0, 3e-9
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=_ice_volume_salt_config(
            slope,
            constant_unaccounted_salting=constant,
            weight_by_sea_surface_fraction=True,
        )
    )
    corrector = config.get_corrector(dataset_info)

    def salinity(value):
        so = value + torch.rand(2, nlat, nlon, dtype=torch.float64, device=DEVICE)
        return so.where(ocean_mask.to(DEVICE) > 0, float("nan"))

    input_ice = torch.rand(nlat, nlon, dtype=torch.float64) * 1e10 * ice_mask
    gen_ice = input_ice + torch.rand(nlat, nlon, dtype=torch.float64) * 1e9 * ice_mask
    input_so, gen_so = salinity(34.0), salinity(35.0)
    input_data = {
        "so_0": input_so[0],
        "so_1": input_so[1],
        "sea_ice_volume": input_ice.to(DEVICE),
    }
    gen_data = {
        "so_0": gen_so[0],
        "so_1": gen_so[1],
        "sea_ice_volume": gen_ice.to(DEVICE),
    }
    forcing_data = {"sea_surface_fraction": sea_surface_fraction.to(DEVICE)}
    corrected = corrector(input_data, gen_data, forcing_data, None).corrected

    sea_surface_area = ocean_cell_area * sea_surface_fraction.to(DEVICE)
    expected_change = slope * float(
        (gen_ice - input_ice).sum()
    ) + constant * dataset_info.timestep.total_seconds() * float(sea_surface_area.sum())
    torch.testing.assert_close(
        _total_salt_content(corrected, sea_surface_area, layer_thickness),
        _total_salt_content(input_data, sea_surface_area, layer_thickness)
        + expected_change,
        rtol=1e-12,
        atol=0.0,
    )
    # which is not the budget met by the unweighted content
    assert not torch.allclose(
        _total_salt_content(corrected, ocean_cell_area, layer_thickness),
        _total_salt_content(input_data, ocean_cell_area, layer_thickness)
        + expected_change,
    )


def test_ocean_salt_content_correction_float64_meets_budget_for_float32_state():
    # At realistic magnitudes (column salt ~1e5 psu m, budget term ~3e-2 psu m
    # per unit area) the expected change is a couple of float32 epsilons of the
    # total. With use_float64 it is still met closely, and the corrected
    # salinity keeps the state's float32 dtype.
    torch.manual_seed(0)
    nlat, nlon = 32, 64
    ocean_mask = torch.ones(nlat, nlon)
    layer_thickness = (1000.0, 3000.0)
    dataset_info = _salt_dataset_info(ocean_mask, layer_thickness)
    ocean_cell_area = _ocean_cell_area_m2(ocean_mask)
    slope = 40.0
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=_ice_volume_salt_config(slope, use_float64=True)
    )
    corrector = config.get_corrector(dataset_info)
    gen_ice = torch.zeros(nlat, nlon, device=DEVICE)
    gen_ice[0, :] = 3e-2 * float(ocean_cell_area.sum()) / slope / nlon
    input_so = 35.0 + torch.rand(2, nlat, nlon, device=DEVICE)
    # a slightly biased prediction, so the ratio has work to do
    gen_so = input_so + 1e-3 + 1e-4 * torch.randn(2, nlat, nlon, device=DEVICE)
    input_data = {
        "so_0": input_so[0],
        "so_1": input_so[1],
        "sea_ice_volume": torch.zeros(nlat, nlon, device=DEVICE),
    }
    gen_data = {"so_0": gen_so[0], "so_1": gen_so[1], "sea_ice_volume": gen_ice}
    corrected = corrector(input_data, gen_data, {}, None).corrected

    assert corrected["so_0"].dtype == torch.float32
    expected_change = slope * float(gen_ice.double().sum())
    miss = _total_salt_content(corrected, ocean_cell_area, layer_thickness) - (
        _total_salt_content(input_data, ocean_cell_area, layer_thickness)
        + expected_change
    )
    assert abs(float(miss)) < 0.05 * expected_change


def _ssh_budget_corrector(
    dataset_info: DatasetInfo, reference_salinity_psu: float = 35.0
) -> CorrectorABC:
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=OceanSaltContentBudgetConfig(
            method="scaled_salinity",
            budget_config=SeaSurfaceHeightSaltBudgetConfig(
                reference_salinity_psu=reference_salinity_psu,
                include_brine_rejection=False,
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
    ocean_mask, _ = _salt_ocean_and_ice_masks()
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
    ocean_mask, _ = _salt_ocean_and_ice_masks()
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
        sea_ice_fraction_threshold=0.0,
        full_ice_cover_threshold=None,
        sea_ice_salt_flux=None,
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
    ocean_mask, _ = _salt_ocean_and_ice_masks()
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


@pytest.mark.parametrize(
    "budget_config, expected",
    [
        pytest.param(None, None, id="none"),
        pytest.param(
            {"type": "ice_volume", "slope_psu": 41.6},
            IceVolumeSaltBudgetConfig(slope_psu=41.6),
            id="ice_volume",
        ),
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
        pytest.param(
            {"type": "water_flux"},
            WaterFluxSaltBudgetConfig(),
            id="water_flux",
        ),
        pytest.param(
            {
                "type": "water_flux",
                "use_precipitation_minus_evaporation_over_open_water": True,
                "regimes": {"coast_sea_surface_fraction_threshold": 0.9},
            },
            WaterFluxSaltBudgetConfig(
                use_precipitation_minus_evaporation_over_open_water=True,
                regimes=WaterFluxRegimesConfig(
                    coast_sea_surface_fraction_threshold=0.9
                ),
            ),
            id="water_flux_with_terms",
        ),
    ],
)
def test_ocean_salt_content_budget_config_from_state(budget_config, expected):
    state = {"method": "scaled_salinity", "budget_config": budget_config}
    config = OceanCorrectorConfig.from_state({"ocean_salt_content_correction": state})
    assert config.ocean_salt_content_correction is not None
    assert config.ocean_salt_content_correction.budget_config == expected


@pytest.mark.parametrize(
    "slope, expected_budget",
    [
        pytest.param(0.0, None, id="zero_slope"),
        pytest.param(39.617, IceVolumeSaltBudgetConfig(39.617), id="slope"),
    ],
)
def test_ocean_salt_content_budget_config_loads_deprecated_ice_volume_slope(
    slope, expected_budget
):
    # Configs from before budget_config set ice_volume_salt_slope_psu, fitted
    # to the unweighted salt content, and must load with the same behavior.
    state = {
        "ocean_salt_content_correction": {
            "method": "scaled_salinity",
            "ice_volume_salt_slope_psu": slope,
            "constant_unaccounted_salting": 5e-11,
        }
    }
    with pytest.warns(DeprecationWarning, match="ice_volume_salt_slope_psu"):
        config = OceanCorrectorConfig.from_state(state)
    assert config.ocean_salt_content_correction == OceanSaltContentBudgetConfig(
        method="scaled_salinity",
        budget_config=expected_budget,
        constant_unaccounted_salting=5e-11,
        weight_by_sea_surface_fraction=False,
    )
    # the input state is not modified
    assert "ice_volume_salt_slope_psu" in state["ocean_salt_content_correction"]


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
                    include_brine_rejection=False,
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


@pytest.mark.parametrize("slope, raises", [(0.0, False), (40.0, True)])
def test_ocean_salt_content_correction_spatial_parallelism(slope: float, raises: bool):
    # The ice-volume total is a plain sum over the local grid, so the ice
    # volume budget must fail at build time under spatial parallelism instead of
    # silently using a per-rank total. Without a budget, the area-weighted sums
    # already reduce across ranks.
    dataset_info = _salt_dataset_info(torch.ones(4, 8), (1000.0, 3000.0))
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=_ice_volume_salt_config(slope)
    )
    with patch.object(
        Distributed,
        "has_spatial_parallelism",
        new_callable=PropertyMock,
        return_value=True,
    ):
        if raises:
            with pytest.raises(NotImplementedError, match="local spatial chunk"):
                config.get_corrector(dataset_info)
        else:
            config.get_corrector(dataset_info)


_WATER_FLUX_LAYERS = (10.0, 20.0)


@dataclasses.dataclass
class _WaterFluxSaltState:
    input_data: dict[str, torch.Tensor]
    gen_data: dict[str, torch.Tensor]
    forcing_data: dict[str, torch.Tensor]
    ocean_mask: torch.Tensor
    ice_mask: torch.Tensor
    timestep_seconds: float

    @property
    def sea_surface_area(self) -> torch.Tensor:
        return (
            _ocean_cell_area_m2(self.ocean_mask)
            * self.forcing_data["sea_surface_fraction"]
        )

    def regimes(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """float64 reference ice-covered, coast and open-water masks for the
        default thresholds."""
        ice_covered = (self.input_data["ocean_sea_ice_fraction"].nan_to_num() > 0) | (
            self.gen_data["ocean_sea_ice_fraction"].nan_to_num() > 0
        )
        coast = self.forcing_data["sea_surface_fraction"] < 1.0
        return ice_covered, coast, ~ice_covered & ~coast


def _water_flux_salt_state() -> _WaterFluxSaltState:
    """A float64 state on the grid of ``_salt_ocean_and_ice_masks`` with every
    regime: ice in the rows of the ice mask (and one cell where ice forms over
    the step, outside the sea_ice_volume mask), coast in the partly-land second
    row, and open water elsewhere. wfo is NaN over land and sfdsi NaN where
    there is never ice; sea_ice_volume outside its mask is unconstrained and
    large."""
    torch.manual_seed(0)
    ocean_mask, ice_mask = _salt_ocean_and_ice_masks()
    ocean = ocean_mask.to(DEVICE) > 0
    ice = ice_mask.to(DEVICE) > 0
    nan = float("nan")

    def rand() -> torch.Tensor:
        return torch.rand(ocean_mask.shape, dtype=torch.float64, device=DEVICE)

    sea_surface_fraction = torch.ones_like(rand()) * ocean
    sea_surface_fraction[1] = (0.2 + 0.7 * rand()[1]) * ocean[1]
    input_ice_fraction = (0.1 + 0.9 * rand()) * ice
    gen_ice_fraction = (0.1 + 0.9 * rand()) * ice
    gen_ice_fraction[2, 5] = 0.3  # ice forms over the step
    input_ice = (rand() * 1e10).where(ice, nan)
    gen_ice = torch.where(ice, input_ice + rand() * 1e9, rand() * 1e12)
    input_data = {
        "so_0": (34.0 + rand()).where(ocean, nan),
        "so_1": (34.0 + rand()).where(ocean, nan),
        "sea_ice_volume": input_ice,
        "ocean_sea_ice_fraction": input_ice_fraction.where(ocean, nan),
        "land_fraction": 1.0 - sea_surface_fraction,
    }
    gen_data = {
        "so_0": (35.0 + rand()).where(ocean, nan),
        "so_1": (35.0 + rand()).where(ocean, nan),
        "sea_ice_volume": gen_ice,
        "ocean_sea_ice_fraction": gen_ice_fraction.where(ocean, nan),
        "wfo": (2e-5 * (rand() - 0.3)).where(ocean, nan),
        "sfdsi": (1e-6 * (rand() - 0.5)).where(ice, nan),
    }
    forcing_data = {
        "sea_surface_fraction": sea_surface_fraction,
        "PRATEsfc": 3e-5 * rand(),
        "LHTFLsfc": 100.0 * rand(),
    }
    timestep = _salt_dataset_info(ocean_mask, _WATER_FLUX_LAYERS).timestep
    return _WaterFluxSaltState(
        input_data,
        gen_data,
        forcing_data,
        ocean_mask,
        ice_mask,
        timestep.total_seconds(),
    )


def _correct_with_water_flux_budget(
    state: _WaterFluxSaltState, water_flux_budget: dict
) -> TensorMapping:
    config = OceanCorrectorConfig.from_state(
        {
            "ocean_salt_content_correction": {
                "method": "scaled_salinity",
                "budget_config": {"type": "water_flux", **water_flux_budget},
            }
        }
    )
    corrector = config.get_corrector(
        _salt_dataset_info(state.ocean_mask, _WATER_FLUX_LAYERS, state.ice_mask)
    )
    return corrector(
        state.input_data, state.gen_data, state.forcing_data, None
    ).corrected


def _salt_content_change(state: _WaterFluxSaltState, corrected: TensorMapping) -> float:
    area = state.sea_surface_area
    return float(
        _total_salt_content(corrected, area, _WATER_FLUX_LAYERS)
        - _total_salt_content(state.input_data, area, _WATER_FLUX_LAYERS)
    )


def _flux_salt_change(
    state: _WaterFluxSaltState,
    water: torch.Tensor,
    salt: torch.Tensor,
    reference_salinity_psu: float = 35.0,
) -> float:
    """float64 reference budget (DT / rho_0) * sum((-S_ref * water + 1000 *
    salt) * ssf * A) in psu m**3, for per-ocean-area fluxes."""
    area = state.sea_surface_area
    flux = -reference_salinity_psu * water.nan_to_num() + 1000.0 * salt.nan_to_num()
    total = float((flux * area).sum())
    return state.timestep_seconds / 1035.0 * total


def _assert_salt_change(state, corrected, expected_change):
    # the change is the difference of two totals ~1e6 times larger, so float64
    # rounding of the totals limits the relative precision of the change
    torch.testing.assert_close(
        _salt_content_change(state, corrected), expected_change, rtol=1e-9, atol=0.0
    )
    # by one ratio applied to every level
    ratio = corrected["so_0"] / state.gen_data["so_0"]
    torch.testing.assert_close(
        corrected["so_1"], state.gen_data["so_1"] * ratio, equal_nan=True
    )


@pytest.mark.parametrize("include_brine_rejection", [True, False])
def test_water_flux_salt_budget_from_wfo(include_brine_rejection):
    # With no terms, the budget is the flux form with the predicted wfo and
    # sfdsi everywhere, NaN counting as zero.
    state = _water_flux_salt_state()
    corrected = _correct_with_water_flux_budget(
        state,
        {
            "include_brine_rejection": include_brine_rejection,
            "reference_salinity_psu": 34.0,
        },
    )
    sfdsi = state.gen_data["sfdsi"] if include_brine_rejection else torch.zeros(1)
    _assert_salt_change(
        state,
        corrected,
        _flux_salt_change(state, state.gen_data["wfo"], sfdsi, 34.0),
    )


@pytest.mark.parametrize("name", ["wfo", "sfdsi"])
def test_water_flux_salt_budget_does_not_fall_back_to_forcing(name):
    # A flux missing from the generated data is not taken from the forcing
    # data, where it would be the target.
    state = _water_flux_salt_state()
    state.forcing_data[name] = state.gen_data.pop(name)
    with pytest.raises(ValueError, match=f"needs {name} in the generated data"):
        _correct_with_water_flux_budget(state, {})


def test_water_flux_salt_budget_fluxes_from_forcing():
    state = _water_flux_salt_state()
    expected = _salt_content_change(state, _correct_with_water_flux_budget(state, {}))
    for name in ("wfo", "sfdsi"):
        state.forcing_data[name] = state.gen_data.pop(name)
    corrected = _correct_with_water_flux_budget(state, {"fluxes_from_forcing": True})
    torch.testing.assert_close(_salt_content_change(state, corrected), expected)


def test_water_flux_salt_budget_terms_replace_wfo_in_their_regimes():
    # wfo counts only where no enabled term covers the cell: changing it over
    # open water (P - E on) or under ice (sea ice mass on) has no effect, while
    # changing it at the coast does.
    state = _water_flux_salt_state()
    ice_covered, coast, open_water = state.regimes()
    assert ice_covered.any() and coast.any() and open_water.any()
    budget = {
        "use_precipitation_minus_evaporation_over_open_water": True,
        "use_sea_ice_mass_change_under_ice": True,
    }
    baseline = _salt_content_change(
        state, _correct_with_water_flux_budget(state, budget)
    )
    wfo = state.gen_data["wfo"]
    for regime, counts in [(open_water, False), (ice_covered, False), (coast, True)]:
        state.gen_data["wfo"] = torch.where(regime, wfo + 1e-4, wfo)
        change = _salt_content_change(
            state, _correct_with_water_flux_budget(state, budget)
        )
        state.gen_data["wfo"] = wfo
        if counts:
            assert abs(change - baseline) > 1e-6 * abs(baseline)
        else:
            torch.testing.assert_close(change, baseline)


def _counted_ice_mass_change_kg_per_s(
    state: _WaterFluxSaltState, region: torch.Tensor
) -> float:
    """float64 reference rho_ice * sum(delta sea_ice_volume) / DT over the
    region, inside the sea_ice_volume mask only."""
    counted = region & (state.ice_mask.to(DEVICE) > 0)
    ice_volume_change = (
        state.gen_data["sea_ice_volume"] - state.input_data["sea_ice_volume"]
    )
    return 905.0 * float(ice_volume_change[counted].sum()) / state.timestep_seconds


@pytest.mark.parametrize(
    "sea_ice_salinity_psu, use_computed_brine_under_ice",
    [(None, False), (4.0, False), (4.0, True)],
)
def test_water_flux_salt_budget_with_terms(
    sea_ice_salinity_psu, use_computed_brine_under_ice
):
    # P - E over open water, the sea ice mass change under ice (inside the
    # sea_ice_volume mask only: the large prediction outside it would miss the
    # budget if counted) and wfo at the coast; the predicted sfdsi, replaced by
    # the computed sea ice salt flux under ice only when asked for.
    state = _water_flux_salt_state()
    corrected = _correct_with_water_flux_budget(
        state,
        {
            "use_precipitation_minus_evaporation_over_open_water": True,
            "use_sea_ice_mass_change_under_ice": True,
            "sea_ice_salinity_psu": sea_ice_salinity_psu,
            "use_computed_brine_under_ice": use_computed_brine_under_ice,
        },
    )
    ice_covered, _, open_water = state.regimes()
    forcing, gen = state.forcing_data, state.gen_data
    precipitation_minus_evaporation = forcing["PRATEsfc"] - forcing["LHTFLsfc"] / 2.5e6
    water = torch.where(open_water, precipitation_minus_evaporation, gen["wfo"])
    water = torch.where(ice_covered, 0.0, water)
    ice_mass_change = _counted_ice_mass_change_kg_per_s(state, ice_covered)
    salt = gen["sfdsi"]
    ice_salt_change = 0.0
    if use_computed_brine_under_ice:
        salt = torch.where(ice_covered, 0.0, salt)
        ice_salt_change = sea_ice_salinity_psu * ice_mass_change
    expected = _flux_salt_change(state, water, salt) + (
        state.timestep_seconds / 1035.0
    ) * (35.0 * ice_mass_change - ice_salt_change)
    _assert_salt_change(state, corrected, expected)


@pytest.mark.parametrize(
    "budget_config, expected",
    [
        pytest.param({"type": "ice_volume", "slope_psu": 40.0}, set(), id="ice_volume"),
        pytest.param({"type": "water_flux"}, set(), id="water_flux_generated"),
        pytest.param(
            {"type": "water_flux", "fluxes_from_forcing": True},
            {"wfo", "sfdsi"},
            id="water_flux_forcing",
        ),
        pytest.param(
            {
                "type": "water_flux",
                "fluxes_from_forcing": True,
                "include_brine_rejection": False,
            },
            {"wfo"},
            id="water_flux_forcing_no_brine",
        ),
        pytest.param(
            {"type": "sea_surface_height"}, set(), id="sea_surface_height_generated"
        ),
        pytest.param(
            {"type": "sea_surface_height", "fluxes_from_forcing": True},
            {"sfdsi"},
            id="sea_surface_height_forcing",
        ),
    ],
)
def test_salt_budget_forcing_names(budget_config, expected):
    # Fluxes read from the forcing data are requested from the step, which
    # would otherwise not provide output variables such as wfo and sfdsi.
    corrector = CorrectorSelector(
        "ocean_corrector",
        {
            "ocean_salt_content_correction": {
                "method": "scaled_salinity",
                "budget_config": budget_config,
            }
        },
    )
    assert corrector.forcing_names == expected


def test_water_flux_salt_budget_full_ice_cover_threshold():
    # With a full ice cover threshold, the sea ice mass change replaces wfo
    # only where the ice fraction reaches it at both steps; the other
    # ice-covered cells keep wfo and are still left out of P - E.
    state = _water_flux_salt_state()
    for data in (state.input_data, state.gen_data):
        data["ocean_sea_ice_fraction"][0, :4] = 1.0
    state.input_data["ocean_sea_ice_fraction"][0, 4] = 1.0  # at one step only
    corrected = _correct_with_water_flux_budget(
        state,
        {
            "use_precipitation_minus_evaporation_over_open_water": True,
            "use_sea_ice_mass_change_under_ice": True,
            "regimes": {"full_ice_cover_threshold": 1.0},
        },
    )
    _, _, open_water = state.regimes()
    full_ice_cover = torch.zeros_like(open_water)
    full_ice_cover[0, :4] = True
    forcing, gen = state.forcing_data, state.gen_data
    precipitation_minus_evaporation = forcing["PRATEsfc"] - forcing["LHTFLsfc"] / 2.5e6
    water = torch.where(open_water, precipitation_minus_evaporation, gen["wfo"])
    water = torch.where(full_ice_cover, 0.0, water)
    ice_mass_change = _counted_ice_mass_change_kg_per_s(state, full_ice_cover)
    expected = _flux_salt_change(state, water, gen["sfdsi"]) + (
        state.timestep_seconds / 1035.0
    ) * (35.0 * ice_mass_change)
    _assert_salt_change(state, corrected, expected)


def test_water_flux_salt_budget_computes_sfdsi_if_not_predicted():
    # Without a predicted sfdsi, the sea ice salt flux is computed from the ice
    # volume change everywhere inside the sea_ice_volume mask.
    state = _water_flux_salt_state()
    del state.gen_data["sfdsi"]
    # a cell in the mask with no ice fraction at either step still counts
    for data in (state.input_data, state.gen_data):
        data["ocean_sea_ice_fraction"][0, 0] = 0.0
    corrected = _correct_with_water_flux_budget(state, {"sea_ice_salinity_psu": 3.0})
    everywhere = torch.ones_like(state.ice_mask, dtype=torch.bool, device=DEVICE)
    ice_salt_change = 3.0 * _counted_ice_mass_change_kg_per_s(state, everywhere)
    expected = (
        _flux_salt_change(state, state.gen_data["wfo"], torch.zeros(1))
        - state.timestep_seconds / 1035.0 * ice_salt_change
    )
    _assert_salt_change(state, corrected, expected)


def test_water_flux_salt_budget_signs():
    # From a persisted salinity, fresh water into the ocean lowers the salt
    # content, salt from the sea ice raises it, and freezing ice raises it.
    state = _water_flux_salt_state()
    state.gen_data.update(so_0=state.input_data["so_0"], so_1=state.input_data["so_1"])

    def content_change(wfo: float, sfdsi: float, budget: dict) -> float:
        state.gen_data["wfo"] = torch.full_like(state.gen_data["wfo"], wfo)
        state.gen_data["sfdsi"] = torch.full_like(state.gen_data["sfdsi"], sfdsi)
        corrected = _correct_with_water_flux_budget(state, budget)
        return _salt_content_change(state, corrected)

    assert content_change(wfo=1e-5, sfdsi=0.0, budget={}) < 0.0
    assert content_change(wfo=0.0, sfdsi=1e-6, budget={}) > 0.0
    # the generated ice volume is larger inside the mask
    assert (
        content_change(
            wfo=0.0, sfdsi=0.0, budget={"use_sea_ice_mass_change_under_ice": True}
        )
        > 0.0
    )


@pytest.mark.parametrize(
    "budget, data_name, missing, match",
    [
        ({}, "gen_data", "wfo", "needs wfo"),
        ({}, "gen_data", "sfdsi", "needs sfdsi"),
        (
            {"use_precipitation_minus_evaporation_over_open_water": True},
            "forcing_data",
            "PRATEsfc",
            "PRATEsfc",
        ),
        (
            {"use_sea_ice_mass_change_under_ice": True},
            "gen_data",
            "sea_ice_volume",
            "sea_ice_volume",
        ),
    ],
)
def test_water_flux_salt_budget_missing_field_raises(budget, data_name, missing, match):
    # A flux the configuration needs must be present, not silently read as zero.
    state = _water_flux_salt_state()
    del getattr(state, data_name)[missing]
    with pytest.raises(ValueError, match=match):
        _correct_with_water_flux_budget(state, budget)


@pytest.mark.parametrize(
    "salt_config, match",
    [
        pytest.param(
            {
                "budget_config": {
                    "type": "water_flux",
                    "sea_ice_salinity_psu": 4.0,
                    "include_brine_rejection": False,
                }
            },
            "requires include_brine_rejection",
            id="ice_salinity_without_brine_rejection",
        ),
        pytest.param(
            {
                "budget_config": {
                    "type": "water_flux",
                    "use_computed_brine_under_ice": True,
                }
            },
            "requires sea_ice_salinity_psu",
            id="computed_brine_without_ice_salinity",
        ),
        pytest.param(
            {
                "budget_config": {
                    "type": "water_flux",
                    "use_sea_ice_mass_change_under_ice": True,
                    "sea_ice_density_kg_m3": 0.0,
                }
            },
            "sea_ice_density_kg_m3 must be positive",
            id="nonpositive_ice_density",
        ),
        pytest.param(
            {
                "budget_config": {
                    "type": "water_flux",
                    "regimes": {"full_ice_cover_threshold": 1.0},
                }
            },
            "full_ice_cover_threshold requires",
            id="full_ice_cover_without_ice_terms",
        ),
        pytest.param(
            {
                "budget_config": {
                    "type": "water_flux",
                    "use_sea_ice_mass_change_under_ice": True,
                    "regimes": {
                        "sea_ice_fraction_threshold": 0.5,
                        "full_ice_cover_threshold": 0.5,
                    },
                }
            },
            "must exceed sea_ice_fraction_threshold",
            id="full_ice_cover_not_above_ice_covered",
        ),
        pytest.param(
            {
                "budget_config": {"type": "water_flux"},
                "weight_by_sea_surface_fraction": False,
            },
            "requires weight_by_sea_surface_fraction",
            id="unweighted_content",
        ),
    ],
)
def test_water_flux_salt_budget_config_validation(salt_config, match):
    salt_state = {"method": "scaled_salinity", **salt_config}
    with pytest.raises(ValueError, match=match):
        OceanCorrectorConfig.from_state({"ocean_salt_content_correction": salt_state})


@pytest.mark.parametrize(
    "use_sea_ice_mass_change_under_ice, raises", [(False, False), (True, True)]
)
def test_water_flux_salt_budget_spatial_parallelism(
    use_sea_ice_mass_change_under_ice, raises
):
    # The sea ice mass term sums sea_ice_volume over the local grid, so it must
    # fail at build time under spatial parallelism; the other terms reduce
    # across ranks.
    dataset_info = _salt_dataset_info(torch.ones(4, 8), (1000.0, 3000.0))
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=OceanSaltContentBudgetConfig(
            method="scaled_salinity",
            budget_config=WaterFluxSaltBudgetConfig(
                use_sea_ice_mass_change_under_ice=use_sea_ice_mass_change_under_ice
            ),
        )
    )
    with patch.object(
        Distributed,
        "has_spatial_parallelism",
        new_callable=PropertyMock,
        return_value=True,
    ):
        if raises:
            with pytest.raises(NotImplementedError, match="local spatial chunk"):
                config.get_corrector(dataset_info)
        else:
            config.get_corrector(dataset_info)


def _ssh_state_with_sea_ice() -> _WaterFluxSaltState:
    """The water flux test state with a sea surface height in place of wfo."""
    state = _water_flux_salt_state()
    ocean = state.ocean_mask.to(DEVICE) > 0
    shape = state.ocean_mask.shape
    input_ssh = 0.5 * torch.randn(shape, dtype=torch.float64, device=DEVICE)
    gen_ssh = input_ssh + 1e-3 * torch.randn(shape, dtype=torch.float64, device=DEVICE)
    state.input_data["SSH"] = input_ssh.where(ocean, float("nan"))
    state.gen_data["SSH"] = gen_ssh.where(ocean, float("nan"))
    del state.gen_data["wfo"]
    return state


def _correct_with_ssh_budget(
    state: _WaterFluxSaltState, ssh_budget: dict
) -> TensorMapping:
    config = OceanCorrectorConfig.from_state(
        {
            "ocean_salt_content_correction": {
                "method": "scaled_salinity",
                "budget_config": {"type": "sea_surface_height", **ssh_budget},
            }
        }
    )
    corrector = config.get_corrector(
        _salt_dataset_info(state.ocean_mask, _WATER_FLUX_LAYERS, state.ice_mask)
    )
    return corrector(
        state.input_data, state.gen_data, state.forcing_data, None
    ).corrected


def _ssh_height_salt_change(state: _WaterFluxSaltState) -> float:
    """float64 reference -S_ref * sum(delta SSH * ssf * A) in psu m**3."""
    height_change = (state.gen_data["SSH"] - state.input_data["SSH"]).nan_to_num()
    return -35.0 * float((height_change * state.sea_surface_area).sum())


@pytest.mark.parametrize(
    "sea_ice_salinity_psu, use_computed_brine_under_ice",
    [(None, False), (3.0, False), (3.0, True)],
)
def test_sea_surface_height_salt_budget_with_sea_ice_salt_flux(
    sea_ice_salinity_psu, use_computed_brine_under_ice
):
    # The height term plus the predicted sfdsi, replaced by the computed sea
    # ice salt flux under ice only when asked for.
    state = _ssh_state_with_sea_ice()
    corrected = _correct_with_ssh_budget(
        state,
        {
            "sea_ice_salinity_psu": sea_ice_salinity_psu,
            "use_computed_brine_under_ice": use_computed_brine_under_ice,
        },
    )
    ice_covered, _, _ = state.regimes()
    salt = state.gen_data["sfdsi"]
    ice_salt_change = 0.0
    if use_computed_brine_under_ice:
        salt = torch.where(ice_covered, 0.0, salt)
        ice_salt_change = sea_ice_salinity_psu * _counted_ice_mass_change_kg_per_s(
            state, ice_covered
        )
    expected = (
        _ssh_height_salt_change(state)
        + _flux_salt_change(state, torch.zeros(1), salt)
        - state.timestep_seconds / 1035.0 * ice_salt_change
    )
    _assert_salt_change(state, corrected, expected)


def test_sea_surface_height_salt_budget_full_ice_cover_threshold():
    # With a full ice cover threshold, the computed sea ice salt flux replaces
    # sfdsi only where the ice fraction reaches it at both steps.
    state = _ssh_state_with_sea_ice()
    for data in (state.input_data, state.gen_data):
        data["ocean_sea_ice_fraction"][0, :4] = 1.0
    state.input_data["ocean_sea_ice_fraction"][0, 4] = 1.0  # at one step only
    corrected = _correct_with_ssh_budget(
        state,
        {
            "sea_ice_salinity_psu": 3.0,
            "use_computed_brine_under_ice": True,
            "full_ice_cover_threshold": 1.0,
        },
    )
    full_ice_cover = torch.zeros_like(state.ice_mask, dtype=torch.bool, device=DEVICE)
    full_ice_cover[0, :4] = True
    salt = torch.where(full_ice_cover, 0.0, state.gen_data["sfdsi"])
    ice_salt_change = 3.0 * _counted_ice_mass_change_kg_per_s(state, full_ice_cover)
    expected = (
        _ssh_height_salt_change(state)
        + _flux_salt_change(state, torch.zeros(1), salt)
        - state.timestep_seconds / 1035.0 * ice_salt_change
    )
    _assert_salt_change(state, corrected, expected)


def test_sea_surface_height_salt_budget_computes_sfdsi_if_not_predicted():
    state = _ssh_state_with_sea_ice()
    del state.gen_data["sfdsi"]
    corrected = _correct_with_ssh_budget(state, {"sea_ice_salinity_psu": 3.0})
    everywhere = torch.ones_like(state.ice_mask, dtype=torch.bool, device=DEVICE)
    ice_salt_change = 3.0 * _counted_ice_mass_change_kg_per_s(state, everywhere)
    expected = (
        _ssh_height_salt_change(state)
        - state.timestep_seconds / 1035.0 * ice_salt_change
    )
    _assert_salt_change(state, corrected, expected)


def test_sea_surface_height_salt_budget_sfdsi_source():
    # sfdsi missing from the generated data is not taken from the forcing data
    # unless fluxes_from_forcing asks for it.
    state = _ssh_state_with_sea_ice()
    expected = _salt_content_change(state, _correct_with_ssh_budget(state, {}))
    state.forcing_data["sfdsi"] = state.gen_data.pop("sfdsi")
    with pytest.raises(ValueError, match="needs sfdsi in the generated data"):
        _correct_with_ssh_budget(state, {})
    corrected = _correct_with_ssh_budget(state, {"fluxes_from_forcing": True})
    torch.testing.assert_close(_salt_content_change(state, corrected), expected)


@pytest.mark.parametrize(
    "ssh_budget, match",
    [
        pytest.param(
            {"use_computed_brine_under_ice": True},
            "requires sea_ice_salinity_psu",
            id="computed_brine_without_ice_salinity",
        ),
        pytest.param(
            {"full_ice_cover_threshold": 1.0},
            "full_ice_cover_threshold requires use_computed_brine_under_ice",
            id="full_ice_cover_without_computed_brine",
        ),
        pytest.param(
            {
                "sea_ice_salinity_psu": 3.0,
                "use_computed_brine_under_ice": True,
                "sea_ice_fraction_threshold": 0.5,
                "full_ice_cover_threshold": 0.5,
            },
            "must exceed sea_ice_fraction_threshold",
            id="full_ice_cover_not_above_ice_covered",
        ),
    ],
)
def test_sea_surface_height_salt_budget_config_validation(ssh_budget, match):
    with pytest.raises(ValueError, match=match):
        OceanSaltContentBudgetConfig(
            method="scaled_salinity",
            budget_config=SeaSurfaceHeightSaltBudgetConfig(**ssh_budget),
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
        "ocean_salt_content_correction",
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
        ocean_salt_content_correction=OceanSaltContentBudgetConfig(
            method="scaled_salinity",
            budget_config=IceVolumeSaltBudgetConfig(slope_psu=40.0),
            constant_unaccounted_salting=1e-9,
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
        "sea_ice_volume": randoms((n_members, nlat, nlon)).abs(),
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
        "sea_ice_volume": randoms((n_members, nlat, nlon)).abs(),
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
