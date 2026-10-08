import dataclasses
import datetime

import pytest
import torch

from fme import get_device
from fme.core.coordinates import DepthCoordinate
from fme.core.corrector.ocean import (
    OceanCorrectorConfig,
    OceanHeatContentBudgetConfig,
    OpenOceanAnchorConfig,
    SeaIceFractionConfig,
    SeaIceHfdsCorrectionConfig,
    SurfaceEnergyFluxCorrectionConfig,
    ZosGlobalMeanCorrectionConfig,
    _compute_ocean_net_surface_energy_flux,
    block_mean,
    sis2_ice_deficit,
    surface_flux_term,
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


# --- form_A_sis2 hfds correction over sea ice ------------------------------

_LF = 3.34e5
_LV = 2.5e6
_ICE_SHAPE = (2, 4, 3)  # (batch, lat, lon)
_ICE_LAT = torch.tensor([-60.0, -30.0, 30.0, 60.0])
_ICE_AREA = torch.cos(torch.deg2rad(_ICE_LAT)).unsqueeze(-1).expand(4, 3)
_DT = datetime.timedelta(days=5)
_TEMPS = ["T1", "T2", "T3", "T4"]


def _reference_deficit(T: float, S: float) -> float:
    """ecand3_transform.deficit for S > 0 (branches B2, B4)."""
    import math

    t_fr = -0.054 * S
    if T >= t_fr:
        return 4200.0 * (t_fr - T)
    t, a = -T, -t_fr
    return (
        _LF * (1 - a / t) + 2100.0 * (t - a) + (4200.0 - 2100.0) * a * math.log(t / a)
    )


@pytest.mark.parametrize("T", [-20.0, -5.0, -0.1, 0.0, 0.5])
def test_sis2_ice_deficit_matches_reference(T):
    salinity = [0.65, 2.35, 3.03, 3.19]
    expected = sum(_reference_deficit(T, s) for s in salinity) / 4
    layers = [torch.tensor([T], dtype=torch.float64) for _ in salinity]
    torch.testing.assert_close(
        sis2_ice_deficit(layers), torch.tensor([expected], dtype=torch.float64)
    )


def _ice_case(seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(lo, hi):
        return lo + (hi - lo) * torch.rand(_ICE_SHAPE, generator=g)

    ssf = torch.ones(_ICE_SHAPE)
    ssf[:, :, 2] = 0.0  # land column
    ssf[:, 0, 1] = 0.6  # coastal cell on the support
    m_in = r(100.0, 900.0)
    m_gen = r(100.0, 900.0)
    m_in[:, 1, :] = 0.0  # ice-free row at -30
    m_gen[:, 2, 0] = 0.0  # ice at k-1 only: off the support
    m_in = m_in * ssf
    m_gen = m_gen * ssf
    land = 1 - ssf
    sif_in = torch.where(m_in > 0, r(0.3, 1.0), torch.zeros(_ICE_SHAPE))
    sif_gen = torch.where(m_gen > 0, r(0.3, 1.0), torch.zeros(_ICE_SHAPE))
    input_data = {
        "sst": r(271.0, 275.0),
        "land_fraction": land,
        "sea_surface_fraction": ssf,
        "sea_ice_fraction": sif_in,
        "frozen_mass_total_area": m_in,
        **{n: r(-15.0, -1.0) for n in _TEMPS},
    }
    gen_data = {
        "sst": r(271.0, 275.0),
        "hfds_total_area": r(-50.0, 50.0),
        "sea_ice_fraction": sif_gen,
        "frozen_mass_total_area": m_gen,
        **{n: r(-15.0, -1.0) for n in _TEMPS},
        "SNOWFL_total_area": r(0.0, 1e-5),
        "SW_total_area": r(0.0, 100.0),
        "LW_total_area": r(-80.0, -20.0),
        "LH_total_area": r(-5.0, 5.0),
        "SH_total_area": r(-20.0, 20.0),
        "calving_residue_total_area": r(-5.0, 5.0),
    }
    forcing_data = {
        "land_fraction": land,
        "sea_surface_fraction": ssf,
        "hfrunoffds": r(0.0, 1.0),
        **_make_atmos_forcing_data(_ICE_SHAPE, device="cpu"),
    }
    return input_data, gen_data, forcing_data


def _to_device(d):
    return {k: v.to(DEVICE) for k, v in d.items()}


def _ice_corrector(weight="uniform", omit_terms=()):
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_open_ocean",
            sea_ice=SeaIceHfdsCorrectionConfig(
                weight=weight, omit_terms=list(omit_terms)
            ),
        ),
    )
    ops = LatLonOperations(_ICE_AREA)
    return config._build(ops, None, _DT, lat=_ICE_LAT)


def _residual(input_data, gen_data, forcing_data, hfds):
    """Independent restatement of r_c(form_A_sis2), W m-2 per total cell area."""
    ssf = forcing_data["sea_surface_fraction"]

    def D(d):
        Dice = sis2_ice_deficit([d[n] for n in _TEMPS])
        return torch.where(
            d["frozen_mass_total_area"] > 0, Dice, torch.full_like(Dice, _LF)
        )

    g = gen_data
    S = (
        _LF * g["SNOWFL_total_area"]
        - (
            g["SW_total_area"]
            + g["LW_total_area"]
            - g["LH_total_area"]
            - g["SH_total_area"]
        )
        + hfds
        + g["calving_residue_total_area"]
        - forcing_data["hfrunoffds"] * ssf
    )
    storage = (
        g["frozen_mass_total_area"] * D(g)
        - input_data["frozen_mass_total_area"] * D(input_data)
    ) / _DT.total_seconds()
    return S - storage


def _support(input_data, gen_data, forcing_data):
    return (
        (input_data["frozen_mass_total_area"] > 0)
        & (gen_data["frozen_mass_total_area"] > 0)
        & (forcing_data["sea_surface_fraction"] > 0)
    )


@pytest.mark.parametrize("weight", ["uniform", "sea_ice_fraction"])
def test_form_a_sis2_closes_hemispheric_budget(weight):
    input_data, gen_data, forcing_data = _ice_case()
    out = _ice_corrector(weight)(
        _to_device(input_data), _to_device(gen_data), _to_device(forcing_data), None
    ).corrected
    hfds = out["hfds_total_area"].cpu().double()
    r = _residual(
        {k: v.double() for k, v in input_data.items()},
        {k: v.double() for k, v in gen_data.items()},
        {k: v.double() for k, v in forcing_data.items()},
        hfds,
    )
    S = _support(input_data, gen_data, forcing_data)
    area = _ICE_AREA.double() * forcing_data["sea_surface_fraction"].double()
    scale = (area * r.abs()).sum()
    for hemi in (_ICE_LAT < 0, _ICE_LAT > 0):
        on = S & hemi[:, None]
        X = (torch.where(on, r, 0.0) * area).sum(dim=(-2, -1))
        assert torch.all(X.abs() < 1e-5 * scale), X


def test_form_a_sis2_delta_shape_on_support():
    input_data, gen_data, forcing_data = _ice_case()
    out = _ice_corrector("sea_ice_fraction")(
        _to_device(input_data), _to_device(gen_data), _to_device(forcing_data), None
    ).corrected
    delta = out["hfds_total_area"].cpu() - gen_data["hfds_total_area"]
    S = _support(input_data, gen_data, forcing_data)
    sif = gen_data["sea_ice_fraction"]
    for hemi in (_ICE_LAT < 0, _ICE_LAT > 0):
        on = S & hemi[:, None]
        ratio = torch.where(on, delta / sif, torch.nan)
        for b in range(_ICE_SHAPE[0]):
            vals = ratio[b][on[b]]
            torch.testing.assert_close(vals, vals[:1].expand_as(vals))


def test_form_a_sis2_off_support_matches_method():
    input_data, gen_data, forcing_data = _ice_case()
    ice = _ice_corrector()(
        _to_device(input_data), _to_device(gen_data), _to_device(forcing_data), None
    ).corrected
    method_only = (
        OceanCorrectorConfig(
            surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
                method="prescribed_open_ocean"
            ),
        )
        ._build(LatLonOperations(_ICE_AREA), None, _DT)(
            _to_device(input_data), _to_device(gen_data), _to_device(forcing_data), None
        )
        .corrected
    )
    S = _support(input_data, gen_data, forcing_data).to(DEVICE)
    torch.testing.assert_close(
        ice["hfds_total_area"][~S], method_only["hfds_total_area"][~S]
    )
    assert not torch.allclose(
        ice["hfds_total_area"][S], gen_data["hfds_total_area"].to(DEVICE)[S]
    )


def test_form_a_sis2_empty_support_leaves_hfds():
    input_data, gen_data, forcing_data = _ice_case()
    input_data["frozen_mass_total_area"] = torch.zeros(_ICE_SHAPE)
    ice = _ice_corrector()(
        _to_device(input_data), _to_device(gen_data), _to_device(forcing_data), None
    ).corrected
    method_only = (
        OceanCorrectorConfig(
            surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
                method="prescribed_open_ocean"
            ),
        )
        ._build(LatLonOperations(_ICE_AREA), None, _DT)(
            _to_device(input_data), _to_device(gen_data), _to_device(forcing_data), None
        )
        .corrected
    )
    torch.testing.assert_close(ice["hfds_total_area"], method_only["hfds_total_area"])


def test_surface_flux_term_source_priority():
    ssf = torch.full((2, 2), 0.5)
    am4 = _make_atmos_forcing_data((2, 2), device="cpu")
    gen = {"SNOWFL_total_area": torch.full((2, 2), 1e-5)}
    forcing = {"SNOWFL_total_area": torch.full((2, 2), 2e-5), **am4}
    # gen_data first
    torch.testing.assert_close(
        surface_flux_term("lf_snowfl", gen, forcing, ssf),
        _LF * gen["SNOWFL_total_area"],
    )
    # forcing second
    torch.testing.assert_close(
        surface_flux_term("lf_snowfl", {}, forcing, ssf),
        _LF * forcing["SNOWFL_total_area"],
    )
    # AM4 last, per ocean area -> per total cell area
    torch.testing.assert_close(
        surface_flux_term("lf_snowfl", {}, am4, ssf),
        _LF * am4["total_frozen_precipitation_rate"] * ssf,
    )
    f_top_am4 = (
        am4["DSWRFsfc"]
        - am4["USWRFsfc"]
        + am4["DLWRFsfc"]
        - am4["ULWRFsfc"]
        - am4["LHTFLsfc"]
        - am4["SHTFLsfc"]
    )
    torch.testing.assert_close(
        surface_flux_term("minus_f_top", {}, am4, ssf), -f_top_am4 * ssf
    )
    # calving residue from its ocean-flux components
    comps = {
        k: torch.full((2, 2), v)
        for k, v in [("hflso", 3.0), ("evs", 1e-6), ("prsn", 2e-6)]
    }
    torch.testing.assert_close(
        surface_flux_term("calving_residue", {}, comps, ssf),
        (-3.0 + _LV * 1e-6 - _LF * 2e-6) * ssf,
    )
    with pytest.raises(KeyError, match="minus_hfrunoffds"):
        surface_flux_term("minus_hfrunoffds", {}, am4, ssf)


def test_am4_precipitation_heat_counts_snow_once():
    """PRATEsfc is total (liquid + frozen) precipitation."""
    cp, t_f = 3992.0, 273.15
    ssf = torch.full((2, 2), 0.5)
    am4 = _make_atmos_forcing_data((2, 2), device="cpu")
    sst = torch.full((2, 2), 300.0)
    p_h = cp * (am4["PRATEsfc"] - am4["LHTFLsfc"] / _LV) * (sst - t_f)
    torch.testing.assert_close(
        surface_flux_term("precipitation_heat", {}, am4, ssf, sst=sst), p_h * ssf
    )
    f_top = (
        am4["DSWRFsfc"]
        - am4["USWRFsfc"]
        + am4["DLWRFsfc"]
        - am4["ULWRFsfc"]
        - am4["LHTFLsfc"]
        - am4["SHTFLsfc"]
    )
    torch.testing.assert_close(
        _compute_ocean_net_surface_energy_flux(am4, sst),
        f_top - _LF * am4["total_frozen_precipitation_rate"] + p_h,
    )


def test_form_a_sis2_am4_fallback_and_omit_terms():
    input_data, gen_data, forcing_data = _ice_case()
    for n in [
        "SNOWFL_total_area",
        "SW_total_area",
        "LW_total_area",
        "LH_total_area",
        "SH_total_area",
    ]:
        del gen_data[n]
    del forcing_data["hfrunoffds"]
    args = (
        _to_device(input_data),
        _to_device(gen_data),
        _to_device(forcing_data),
        None,
    )
    with pytest.raises(KeyError, match="minus_hfrunoffds"):
        _ice_corrector()(*args)
    out = _ice_corrector(omit_terms=["minus_hfrunoffds"])(*args).corrected
    assert torch.isfinite(out["hfds_total_area"]).all()


def test_sea_ice_hfds_config_from_state():
    state = {
        "surface_energy_flux_correction": {
            "method": "prescribed_open_ocean",
            "sea_ice": {"method": "form_A_sis2", "weight": "uniform"},
        }
    }
    config = OceanCorrectorConfig.from_state(state)
    sef = config.surface_energy_flux_correction
    assert sef is not None and sef.sea_ice is not None
    assert sef.sea_ice.ice_layer_temperature_names == _TEMPS


# --- AM4-anchored open-ocean arm (issue 25 gen_shift_block{n}) --------------

_OO_SHAPE = (2, 6, 6)
_OO_LAT = torch.linspace(-75.0, 75.0, 6)
_OO_AREA = torch.cos(torch.deg2rad(_OO_LAT)).unsqueeze(-1).expand(6, 6)
_Q_O = ["hfds", "calving_residue", "minus_hfrunoffds"]
_Q_A = ["f_top", "minus_lf_snowfl", "precipitation_heat"]


def _oo_case(seed=1):
    g = torch.Generator().manual_seed(seed)

    def r(lo, hi):
        return lo + (hi - lo) * torch.rand(_OO_SHAPE, generator=g)

    ssf = torch.ones(_OO_SHAPE)
    ssf[:, :, 5] = 0.0  # land column
    ssf[:, 2, 4] = 0.5  # coastal cell
    land = 1 - ssf
    sif = torch.zeros(_OO_SHAPE)
    sif[:, 0, :5] = 0.4  # sea ice row
    sif = sif * ssf
    m = torch.where(sif > 0, r(100.0, 900.0), torch.zeros(_OO_SHAPE))
    input_data = {
        "sst": r(271.0, 300.0),
        "land_fraction": land,
        "sea_surface_fraction": ssf,
        "sea_ice_fraction": sif,
        "frozen_mass_total_area": m,
        **{n: r(-15.0, -1.0) for n in _TEMPS},
    }
    gen_data = {
        "sst": r(271.0, 300.0),
        "hfds_total_area": r(-100.0, 100.0) * ssf,
        "sea_ice_fraction": sif,
        "frozen_mass_total_area": m * 1.1,
        **{n: r(-15.0, -1.0) for n in _TEMPS},
        "SNOWFL_total_area": r(0.0, 1e-5) * ssf,
        "SW_total_area": r(0.0, 200.0) * ssf,
        "LW_total_area": r(-80.0, -20.0) * ssf,
        "LH_total_area": r(0.0, 100.0) * ssf,
        "SH_total_area": r(-20.0, 20.0) * ssf,
        "calving_residue_total_area": r(-5.0, 5.0) * ssf,
    }
    forcing_data = {
        "land_fraction": land,
        "sea_surface_fraction": ssf,
        "hfrunoffds": r(0.0, 1.0),
        **{
            k: v * r(0.5, 1.5)
            for k, v in _make_atmos_forcing_data(_OO_SHAPE, device="cpu").items()
        },
    }
    return input_data, gen_data, forcing_data


def _oo_config(q_terms, block_size=3, coastal="method", sea_ice=False):
    return OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed_open_ocean",
            sea_ice=SeaIceHfdsCorrectionConfig() if sea_ice else None,
            open_ocean=OpenOceanAnchorConfig(
                q_terms=q_terms, block_size=block_size, coastal=coastal
            ),
        ),
    )


def _run(config, data, lat=_OO_LAT, area=_OO_AREA):
    return config._build(LatLonOperations(area), None, _DT, lat=lat)(
        *[_to_device(d) for d in data], None
    ).corrected["hfds_total_area"]


def _method_only(data):
    return _run(
        OceanCorrectorConfig(
            surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
                method="prescribed_open_ocean"
            )
        ),
        data,
    )


def _anchored_mask(input_data):
    open_ = (1 - input_data["land_fraction"] - input_data["sea_ice_fraction"]) == 1
    return open_ & (input_data["sea_surface_fraction"] > 0)


@pytest.mark.parametrize("q_terms", [_Q_O, _Q_A], ids=["Q_o", "Q_a"])
def test_open_ocean_anchor_block_integrals_equal_am4(q_terms):
    input_data, gen_data, forcing_data = _oo_case()
    n = 3
    hfds = _run(_oo_config(q_terms, n), (input_data, gen_data, forcing_data))
    hfds = hfds.cpu().double()
    ssf = forcing_data["sea_surface_fraction"].double()
    q = (
        hfds
        + gen_data["calving_residue_total_area"].double()
        - forcing_data["hfrunoffds"].double() * ssf
    )
    am4 = (
        _compute_ocean_net_surface_energy_flux(
            {k: v.double() for k, v in forcing_data.items()},
            input_data["sst"].double(),
        )
        * ssf
    )
    M = _anchored_mask(input_data)
    w = torch.where(M, _OO_AREA.double() * ssf, 0.0)
    shape = (_OO_SHAPE[0], 6 // n, n, 6 // n, n)
    blocks_q = (w * q).reshape(shape).sum(dim=(-3, -1))
    blocks_f = (w * am4).reshape(shape).sum(dim=(-3, -1))
    scale = (w * am4.abs()).sum()
    torch.testing.assert_close(blocks_q, blocks_f, atol=1e-5 * scale, rtol=0)
    torch.testing.assert_close(
        (w * q).sum(dim=(-2, -1)),
        (w * am4).sum(dim=(-2, -1)),
        atol=1e-5 * scale,
        rtol=0,
    )
    # not simply prescribed: the generated pattern survives inside each block
    assert not torch.allclose(q[M], am4[M])


@pytest.mark.parametrize("coastal", ["method", "generated"])
def test_open_ocean_anchor_off_mask_cells(coastal):
    data = _oo_case()
    input_data, gen_data, _ = data
    hfds = _run(_oo_config(_Q_O, coastal=coastal), data).cpu()
    off = ~_anchored_mask(input_data)
    expected = (
        gen_data["hfds_total_area"]
        if coastal == "generated"
        else _method_only(data).cpu()
    )
    torch.testing.assert_close(hfds[off], expected[off])


def test_open_ocean_anchor_excludes_zero_sea_surface_fraction():
    data = _oo_case()
    input_data, _, forcing_data = data
    no_land = torch.zeros(_OO_SHAPE)
    input_data["land_fraction"] = no_land
    forcing_data["land_fraction"] = no_land
    zero_ssf = forcing_data["sea_surface_fraction"] == 0
    assert zero_ssf.any()
    hfds = _run(_oo_config(_Q_O), data).cpu()
    torch.testing.assert_close(hfds[zero_ssf], _method_only(data).cpu()[zero_ssf])


def test_open_ocean_anchor_with_sea_ice_keeps_sea_ice_arm():
    data = _oo_case()
    input_data, gen_data, _ = data
    both = _run(_oo_config(_Q_O, sea_ice=True), data).cpu()
    ice_only = _run(
        OceanCorrectorConfig(
            surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
                method="prescribed_open_ocean", sea_ice=SeaIceHfdsCorrectionConfig()
            )
        ),
        data,
    ).cpu()
    S = (input_data["frozen_mass_total_area"] > 0) & (
        gen_data["frozen_mass_total_area"] > 0
    )
    assert S.any()
    torch.testing.assert_close(both[S], ice_only[S])


@pytest.mark.parametrize("q_terms", [_Q_O, _Q_A], ids=["Q_o", "Q_a"])
def test_open_ocean_anchor_gradients_finite(q_terms):
    input_data, gen_data, forcing_data = _oo_case()
    grads = ["hfds_total_area", "SW_total_area", "calving_residue_total_area"]
    for n in grads:
        gen_data[n] = gen_data[n].clone().requires_grad_(True)
    out = (
        _oo_config(q_terms)
        ._build(LatLonOperations(_OO_AREA), None, _DT, lat=_OO_LAT)(
            input_data, gen_data, forcing_data, None
        )
        .corrected["hfds_total_area"]
    )
    out.pow(2).sum().backward()
    for n in grads:
        g = gen_data[n].grad
        if g is not None:
            assert torch.isfinite(g).all(), n
    assert gen_data["hfds_total_area"].grad is not None


def test_block_mean_rejects_non_dividing_block():
    with pytest.raises(ValueError, match="block_size"):
        block_mean(torch.ones(4, 6), torch.ones(4, 6), 4)


def test_open_ocean_off_leaves_existing_behavior():
    data = _oo_case()
    plain = _method_only(data)
    explicit = _run(
        OceanCorrectorConfig(
            surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
                method="prescribed_open_ocean", open_ocean=None, sea_ice=None
            )
        ),
        data,
    )
    torch.testing.assert_close(plain, explicit)


def test_open_ocean_config_from_state():
    config = OceanCorrectorConfig.from_state(
        {
            "surface_energy_flux_correction": {
                "method": "prescribed_open_ocean",
                "open_ocean": {"q_terms": _Q_A, "block_size": 5},
            }
        }
    )
    sef = config.surface_energy_flux_correction
    assert sef is not None and sef.open_ocean is not None
    oo = sef.open_ocean
    assert (oo.q_terms, oo.block_size, oo.coastal) == (_Q_A, 5, "method")
