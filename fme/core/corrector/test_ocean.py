import datetime
from unittest.mock import patch

import pytest
import torch

from fme import get_device
from fme.core.constants import (
    DENSITY_OF_WATER_CM4,
    REFERENCE_SALINITY_PSU,
    SPECIFIC_HEAT_OF_WATER_CM4,
)
from fme.core.coordinates import DepthCoordinate
from fme.core.corrector.ocean import (
    OceanCorrector,
    OceanCorrectorConfig,
    OceanHeatContentBudgetConfig,
    OceanSaltContentBudgetConfig,
    SeaIceFractionConfig,
    dz_from_idepth,
)
from fme.core.gridded_ops import LatLonOperations
from fme.core.mask_provider import MaskProvider
from fme.core.ocean_data import OceanData
from fme.core.typing_ import TensorMapping

DEVICE = get_device()
IMG_SHAPE = (5, 5)
NZ = 2

_MASK = torch.ones(*IMG_SHAPE, NZ, device=DEVICE)
_LAT, _LON = 2, 2
_MASK[_LAT, _LON, :] = 0.0


class _MockDepth:
    def __len__(self) -> int:
        return len(self.get_idepth())

    def get_mask(self) -> torch.Tensor:
        return _MASK

    def get_mask_level(self, level: int):
        return _MASK.select(-1, level)

    def get_mask_tensor_for(self, name: str) -> torch.Tensor | None:
        if name.endswith("_1"):
            return _MASK.select(-1, 1)
        else:
            return _MASK.select(-1, 0)

    def get_idepth(self) -> torch.Tensor:
        return torch.tensor([0, 5, 15], device=DEVICE)

    def depth_integral(
        self, integrand: torch.Tensor, sea_floor_depth: torch.Tensor | None = None
    ) -> torch.Tensor:
        idepth = self.get_idepth()
        dz = dz_from_idepth(idepth, sea_floor_depth)
        return torch.nansum(_MASK * integrand * dz, dim=-1)

    def build_output_masker(self):
        raise NotImplementedError

    def to(self, device: str) -> "_MockDepth":
        return self


_VERTICAL_COORD = _MockDepth()


def test_ocean_corrector_raises_on_missing_forcing_keys_in_input():
    config = OceanCorrectorConfig()
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = OceanCorrector(config, ops, None, timestep)
    input_data = {"a": torch.zeros(IMG_SHAPE)}
    gen_data = {"a": torch.zeros(IMG_SHAPE)}
    forcing_data = {"b": torch.zeros(IMG_SHAPE), "c": torch.zeros(IMG_SHAPE)}
    with pytest.raises(RuntimeError, match=r"\['b', 'c'\]"):
        corrector(input_data, gen_data, forcing_data)


def test_ocean_corrector_force_positive():
    """"""
    torch.manual_seed(0)
    config = OceanCorrectorConfig(force_positive_names=["so_0", "so_1"])
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = OceanCorrector(config, ops, _VERTICAL_COORD, timestep)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    corrected_gen = corrector(input_data, gen_data, {})
    for name in ["so_0", "so_1"]:
        x = corrected_gen[name].clone()
        x[_LAT, _LON] = 0.0
        assert torch.all(x >= 0.0)


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
    corrector = OceanCorrector(config, ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(input_data, gen_data, next_step_input_data)
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
    corrector = OceanCorrector(config, ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(input_data, gen_data, next_step_input_data)
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
    corrector = OceanCorrector(config, ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {})
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
    corrector = OceanCorrector(config, ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {})
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
    mask_provider = MaskProvider(masks)
    ops = LatLonOperations(torch.ones(size=[3, 3]), mask_provider)

    idepth = torch.tensor([2.5, 10, 20])
    depth_coordinate = DepthCoordinate(idepth, mask)

    sea_surface_fraction = mask[:, :, :, 0]

    input_data_dict = {
        "thetao_0": torch.ones(nsamples, nlat, nlon),
        "thetao_1": torch.ones(nsamples, nlat, nlon),
        "sst": torch.ones(nsamples, nlat, nlon) + 273.15,
        "hfgeou": torch.ones(nsamples, nlat, nlon),
        "sea_surface_fraction": sea_surface_fraction,
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
    corrector = OceanCorrector(config, ops, depth_coordinate, timestep)
    gen_data_corrected_dict = corrector(
        input_data_dict, gen_data_dict, forcing_data_dict
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


def test_ocean_heat_content_correction_dz_3d():
    """Verify OHC correction uses deptho from forcing_data for the depth integral.

    With deptho=15 and idepth=[2.5, 10, 20], the bottom layer effective thickness
    is truncated from 10m to 5m, giving a total column depth of 12.5m instead of
    17.5m.  This changes both the gen OHC and the correction ratio compared to
    the no-deptho case.
    """
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
    mask_provider = MaskProvider(masks)
    ops = LatLonOperations(torch.ones(size=[3, 3]), mask_provider)

    idepth = torch.tensor([2.5, 10, 20])
    depth_coordinate = DepthCoordinate(idepth, mask)

    sea_surface_fraction = mask[:, :, :, 0]
    deptho = torch.full((nsamples, nlat, nlon), 15.0)

    input_data_dict = {
        "thetao_0": torch.ones(nsamples, nlat, nlon),
        "thetao_1": torch.ones(nsamples, nlat, nlon),
        "sst": torch.ones(nsamples, nlat, nlon) + 273.15,
        "hfds": torch.ones(nsamples, nlat, nlon),
        "hfgeou": torch.ones(nsamples, nlat, nlon),
        "sea_surface_fraction": sea_surface_fraction,
        "deptho": deptho,
    }
    gen_data_dict = {
        "thetao_0": torch.ones(nsamples, nlat, nlon) * 2,
        "thetao_1": torch.ones(nsamples, nlat, nlon) * 2,
        "sst": torch.ones(nsamples, nlat, nlon) * 2 + 273.15,
    }
    forcing_data_dict = {
        "hfgeou": torch.ones(nsamples, nlat, nlon),
        "sea_surface_fraction": sea_surface_fraction,
        "deptho": deptho,
    }

    input_data = OceanData(input_data_dict, depth_coordinate)
    gen_data = OceanData({**gen_data_dict, **forcing_data_dict}, depth_coordinate)

    corrector = OceanCorrector(config, ops, depth_coordinate, timestep)
    gen_data_corrected_dict = corrector(
        input_data_dict, gen_data_dict, forcing_data_dict
    )

    input_ohc = input_data.ocean_heat_content.nanmean(dim=(-1, -2), keepdim=True)
    gen_ohc = gen_data.ocean_heat_content.nanmean(dim=(-1, -2), keepdim=True)
    torch.testing.assert_close(gen_ohc, input_ohc * 2, equal_nan=True)

    ohc_change = 2.1 * timestep.total_seconds()
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

    expected_gen_data = OceanData(
        {**expected_gen_data_dict, **forcing_data_dict}, depth_coordinate
    )
    gen_data_corrected = OceanData(gen_data_corrected_dict, depth_coordinate)
    torch.testing.assert_close(
        expected_gen_data.ocean_heat_content,
        gen_data_corrected.ocean_heat_content,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "wfo_type",
    [
        pytest.param("input", id="wfo_in_input"),
        pytest.param("gen", id="wfo_in_gen"),
    ],
)
def test_ocean_salt_content_correction(wfo_type):
    unaccounted_salt_flux = 0.1
    config = OceanCorrectorConfig(
        ocean_salt_content_correction=OceanSaltContentBudgetConfig(
            method="scaled_salinity",
            constant_unaccounted_salt_flux=unaccounted_salt_flux,
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
    mask_provider = MaskProvider(masks)
    ops = LatLonOperations(torch.ones(size=[3, 3]), mask_provider)

    idepth = torch.tensor([2.5, 10, 20])
    depth_coordinate = DepthCoordinate(idepth, mask)

    sea_surface_fraction = mask[:, :, :, 0]

    wfo_value = torch.ones(nsamples, nlat, nlon) * 0.5

    input_data_dict = {
        "so_0": torch.ones(nsamples, nlat, nlon),
        "so_1": torch.ones(nsamples, nlat, nlon),
        "sea_surface_fraction": sea_surface_fraction,
    }
    gen_data_dict = {
        "so_0": torch.ones(nsamples, nlat, nlon) * 2,
        "so_1": torch.ones(nsamples, nlat, nlon) * 2,
    }
    if wfo_type == "gen":
        gen_data_dict["wfo"] = wfo_value
    else:
        input_data_dict["wfo"] = wfo_value
    forcing_data_dict = {
        "sea_surface_fraction": sea_surface_fraction,
    }
    input_data = OceanData(input_data_dict, depth_coordinate)
    gen_data = OceanData(gen_data_dict, depth_coordinate)
    corrector = OceanCorrector(config, ops, depth_coordinate, timestep)
    gen_data_corrected_dict = corrector(
        input_data_dict, gen_data_dict, forcing_data_dict
    )

    input_osc = input_data.ocean_salt_content.nanmean(dim=(-1, -2), keepdim=True)
    gen_osc = gen_data.ocean_salt_content.nanmean(dim=(-1, -2), keepdim=True)
    torch.testing.assert_close(gen_osc, input_osc * 2, equal_nan=True)

    # -reference_salinity * wfo at all non-masked points, plus unaccounted flux;
    # masked points are excluded by the area_weighted_mean so the raw wfo value
    # (0.5 everywhere) directly determines the mean.
    osc_change = (
        -REFERENCE_SALINITY_PSU * 0.5 + unaccounted_salt_flux
    ) * timestep.total_seconds()
    corrector_ratio = (input_osc + osc_change) / gen_osc

    expected_gen_data_dict = {
        key: value * corrector_ratio if key.startswith("so") else value
        for key, value in gen_data_dict.items()
    }

    expected_gen_data = OceanData(expected_gen_data_dict, depth_coordinate)
    gen_data_corrected = OceanData(gen_data_corrected_dict, depth_coordinate)
    torch.testing.assert_close(
        expected_gen_data.ocean_salt_content,
        gen_data_corrected.ocean_salt_content,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
#  Fixtures for MLD-weighted correction tests (4 vertical levels)
# ---------------------------------------------------------------------------

_MLD_NZ = 4
_MLD_NSAMPLES = 2
_MLD_NLAT = 3
_MLD_NLON = 3
_MLD_TIMESTEP = datetime.timedelta(seconds=5 * 24 * 3600)
_MLD_IDEPTH = torch.tensor([0.0, 10.0, 50.0, 200.0, 1000.0])


def _mld_test_fixtures():
    """Build depth coordinate, ops and masks for the 4-level MLD tests."""
    mask = torch.ones(_MLD_NSAMPLES, _MLD_NLAT, _MLD_NLON, _MLD_NZ)
    mask[:, 0, 0, :] = 0.0  # one fully masked column
    masks = {f"mask_{k}": mask[:, :, :, k] for k in range(_MLD_NZ)}
    masks["mask_2d"] = mask[:, :, :, 0]
    mask_provider = MaskProvider(masks)
    ops = LatLonOperations(torch.ones(size=[_MLD_NLAT, _MLD_NLON]), mask_provider)
    depth_coordinate = DepthCoordinate(_MLD_IDEPTH, mask)
    return depth_coordinate, ops, mask


def _mld_input_gen_forcing(mask, *, gen_temp_surface=20.0, gen_temp_deep=5.0):
    """Build input / gen / forcing dicts with a clear pycnocline in gen."""
    shape = (_MLD_NSAMPLES, _MLD_NLAT, _MLD_NLON)
    ssf = mask[:, :, :, 0]

    forcing_data = {
        "hfgeou": torch.ones(shape) * 0.05,
        "sea_surface_fraction": ssf,
        "deptho": torch.full(shape, 1000.0),
    }
    input_data = {
        **{f"thetao_{k}": torch.ones(shape) for k in range(_MLD_NZ)},
        **{f"so_{k}": torch.ones(shape) * 35.0 for k in range(_MLD_NZ)},
        "sst": torch.ones(shape) + 273.15,
        "hfds": torch.ones(shape),
        "wfo": torch.ones(shape) * 0.5,
        **forcing_data,
    }
    gen_data = {
        "thetao_0": torch.ones(shape) * gen_temp_surface,
        "thetao_1": torch.ones(shape) * gen_temp_surface,
        "thetao_2": torch.ones(shape) * gen_temp_deep,
        "thetao_3": torch.ones(shape) * gen_temp_deep,
        **{f"so_{k}": torch.ones(shape) * 35.0 for k in range(_MLD_NZ)},
        "sst": torch.ones(shape) * gen_temp_surface + 273.15,
    }
    return input_data, gen_data, forcing_data


def test_ocean_heat_content_correction_mld():
    depth_coordinate, ops, mask = _mld_test_fixtures()
    input_data, gen_data, forcing_data = _mld_input_gen_forcing(mask)

    config = OceanCorrectorConfig(
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="mixed_layer_depth",
            constant_unaccounted_heating=0.1,
        )
    )
    corrector = OceanCorrector(config, ops, depth_coordinate, _MLD_TIMESTEP)
    corrected = corrector(input_data, gen_data, forcing_data)

    # Check OHC conservation: corrected_ohc == input_ohc + flux * dt
    corrected_ocean = OceanData(corrected, depth_coordinate)
    input_ocean = OceanData(input_data, depth_coordinate)
    corrected_ohc = ops.area_weighted_mean(
        corrected_ocean.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    input_ohc = ops.area_weighted_mean(
        input_ocean.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    ssf = forcing_data["sea_surface_fraction"]
    net_flux = (input_data["hfds"] + forcing_data["hfgeou"]) * ssf
    flux_mean = ops.area_weighted_mean(
        net_flux,
        keepdim=True,
        name="ocean_heat_content",
    )
    expected_ohc = input_ohc + (flux_mean + 0.1) * _MLD_TIMESTEP.total_seconds()
    torch.testing.assert_close(corrected_ohc, expected_ohc, atol=1e-3, rtol=1e-5)

    # Deep layers (below MLD) should be unchanged
    torch.testing.assert_close(corrected["thetao_2"], gen_data["thetao_2"])
    torch.testing.assert_close(corrected["thetao_3"], gen_data["thetao_3"])


def test_ocean_heat_content_correction_mld_geo():
    depth_coordinate, ops, mask = _mld_test_fixtures()
    input_data, gen_data, forcing_data = _mld_input_gen_forcing(mask)

    config = OceanCorrectorConfig(
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="mixed_layer_depth_geo",
        )
    )
    corrector = OceanCorrector(config, ops, depth_coordinate, _MLD_TIMESTEP)
    gen_data_before = {k: v.clone() for k, v in gen_data.items()}
    corrected = corrector(input_data, gen_data, forcing_data)

    # OHC must be conserved globally
    corrected_ocean = OceanData(corrected, depth_coordinate)
    input_ocean = OceanData(input_data, depth_coordinate)
    corrected_ohc = ops.area_weighted_mean(
        corrected_ocean.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    input_ohc = ops.area_weighted_mean(
        input_ocean.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    ssf = forcing_data["sea_surface_fraction"]
    net_flux = (input_data["hfds"] + forcing_data["hfgeou"]) * ssf
    flux_mean = ops.area_weighted_mean(
        net_flux,
        keepdim=True,
        name="ocean_heat_content",
    )
    expected_ohc = input_ohc + flux_mean * _MLD_TIMESTEP.total_seconds()
    torch.testing.assert_close(corrected_ohc, expected_ohc, atol=1e-3, rtol=1e-5)

    # Bottom layer should be warmed by geothermal (not just MLD correction)
    dz_bottom = _MLD_IDEPTH[-1] - _MLD_IDEPTH[-2]  # 800 m
    expected_geo_dT = (
        forcing_data["hfgeou"]
        * ssf
        * _MLD_TIMESTEP.total_seconds()
        / (DENSITY_OF_WATER_CM4 * SPECIFIC_HEAT_OF_WATER_CM4 * dz_bottom)
    )
    bottom_change = corrected["thetao_3"] - gen_data_before["thetao_3"]
    torch.testing.assert_close(
        bottom_change,
        expected_geo_dT,
        atol=1e-6,
        rtol=1e-6,
    )


def test_ocean_salt_content_correction_mld():
    depth_coordinate, ops, mask = _mld_test_fixtures()
    input_data, gen_data, forcing_data = _mld_input_gen_forcing(mask)

    config = OceanCorrectorConfig(
        ocean_salt_content_correction=OceanSaltContentBudgetConfig(
            method="mixed_layer_depth",
        )
    )
    corrector = OceanCorrector(config, ops, depth_coordinate, _MLD_TIMESTEP)
    corrected = corrector(input_data, gen_data, forcing_data)

    # OSC conservation: corrected_osc == input_osc + virtual_salt_flux * dt
    corrected_ocean = OceanData(corrected, depth_coordinate)
    input_ocean = OceanData(input_data, depth_coordinate)
    corrected_osc = ops.area_weighted_mean(
        corrected_ocean.ocean_salt_content,
        keepdim=True,
        name="ocean_salt_content",
    )
    input_osc = ops.area_weighted_mean(
        input_ocean.ocean_salt_content,
        keepdim=True,
        name="ocean_salt_content",
    )
    ssf = forcing_data["sea_surface_fraction"]
    vsf = -REFERENCE_SALINITY_PSU * input_data["wfo"] * ssf
    vsf_mean = ops.area_weighted_mean(
        vsf,
        keepdim=True,
        name="ocean_salt_content",
    )
    expected_osc = input_osc + vsf_mean * _MLD_TIMESTEP.total_seconds()
    torch.testing.assert_close(corrected_osc, expected_osc, atol=1e-3, rtol=1e-5)

    # Deep layers should be unchanged
    torch.testing.assert_close(corrected["so_2"], gen_data["so_2"])
    torch.testing.assert_close(corrected["so_3"], gen_data["so_3"])


def test_mld_weights_reused_from_salt_to_heat():
    """When both salt and heat use MLD, weights computed during salt correction
    are passed to heat correction (avoiding redundant MLD computation)."""
    depth_coordinate, ops, mask = _mld_test_fixtures()
    input_data, gen_data, forcing_data = _mld_input_gen_forcing(mask)

    config = OceanCorrectorConfig(
        ocean_salt_content_correction=OceanSaltContentBudgetConfig(
            method="mixed_layer_depth",
        ),
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="mixed_layer_depth",
        ),
    )
    corrector = OceanCorrector(config, ops, depth_coordinate, _MLD_TIMESTEP)

    with patch(
        "fme.core.corrector.ocean.compute_mld_weights_from_ocean_data",
        wraps=__import__(
            "fme.core.corrector.ocean_mld",
            fromlist=["compute_mld_weights_from_ocean_data"],
        ).compute_mld_weights_from_ocean_data,
    ) as mock_weights:
        corrector(input_data, gen_data, forcing_data)
        # Should be called once (by salt), not twice
        assert mock_weights.call_count == 1

    # Also verify both budgets are conserved simultaneously
    corrected = corrector(input_data, gen_data, forcing_data)
    corrected_ocean = OceanData(corrected, depth_coordinate)
    input_ocean = OceanData(input_data, depth_coordinate)

    corrected_ohc = ops.area_weighted_mean(
        corrected_ocean.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    input_ohc = ops.area_weighted_mean(
        input_ocean.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    ssf = forcing_data["sea_surface_fraction"]
    net_flux = (input_data["hfds"] + forcing_data["hfgeou"]) * ssf
    flux_mean = ops.area_weighted_mean(
        net_flux,
        keepdim=True,
        name="ocean_heat_content",
    )
    expected_ohc = input_ohc + flux_mean * _MLD_TIMESTEP.total_seconds()
    torch.testing.assert_close(corrected_ohc, expected_ohc, atol=1e-3, rtol=1e-5)

    corrected_osc = ops.area_weighted_mean(
        corrected_ocean.ocean_salt_content,
        keepdim=True,
        name="ocean_salt_content",
    )
    input_osc = ops.area_weighted_mean(
        input_ocean.ocean_salt_content,
        keepdim=True,
        name="ocean_salt_content",
    )
    vsf = -REFERENCE_SALINITY_PSU * input_data["wfo"] * ssf
    vsf_mean = ops.area_weighted_mean(vsf, keepdim=True, name="ocean_salt_content")
    expected_osc = input_osc + vsf_mean * _MLD_TIMESTEP.total_seconds()
    torch.testing.assert_close(corrected_osc, expected_osc, atol=1e-3, rtol=1e-5)
