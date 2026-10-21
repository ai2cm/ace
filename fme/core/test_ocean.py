import datetime
from typing import Literal

import dacite
import pytest
import torch

from fme.core.constants import DENSITY_OF_WATER, SPECIFIC_HEAT_OF_WATER
from fme.core.ocean import (
    OceanConfig,
    SlabOceanConfig,
    mixed_layer_temperature_tendency,
)

TIMESTEP = datetime.timedelta(hours=6)


def test_ocean_prescribed():
    config = OceanConfig(surface_temperature_name="sst", ocean_fraction_name="of")
    ocean = config.build(in_names=["sst"], out_names=["sst"], timestep=TIMESTEP)
    target_data = {"sst": torch.tensor([22.0, 25.0]), "of": torch.tensor([0.2, 0.8])}
    input_data = {"sst": torch.tensor([20.0, 21.0]), "foo": torch.tensor([1, 2])}
    gen_data = {"sst": torch.tensor([23.0, 26.0]), "foo": torch.tensor([2, 3])}
    output_data = ocean(input_data, gen_data, target_data)
    expected_output = {"sst": torch.tensor([23.0, 25.0]), "foo": torch.tensor([2, 3])}
    assert set(output_data) == set(expected_output)
    for name in output_data:
        torch.testing.assert_close(output_data[name], expected_output[name])


def test_ocean_slab():
    config = OceanConfig(
        surface_temperature_name="sst",
        ocean_fraction_name="of",
        slab=SlabOceanConfig(
            mixed_layer_depth_name="mld",
            q_flux_name="qf",
        ),
    )
    names_for_net_surface_energy_flux = [
        "DLWRFsfc",
        "ULWRFsfc",
        "DSWRFsfc",
        "USWRFsfc",
        "LHTFLsfc",
        "SHTFLsfc",
    ]
    fluxes = {k: torch.tensor([2.0]) for k in names_for_net_surface_energy_flux}
    expected_net_surface_energy_flux = torch.tensor([-4.0])
    ocean = config.build(in_names=["sst"], out_names=["sst"], timestep=TIMESTEP)
    target_data = {
        "mld": torch.tensor([25.0]),
        "of": torch.tensor([0.8]),
        "qf": torch.tensor([40.0]),
    }
    input_data = {"sst": torch.tensor([20.0])}
    gen_data = {**fluxes, "sst": torch.tensor([25.0])}
    output_data = ocean(input_data, gen_data, target_data)
    expected_sst_tendency = mixed_layer_temperature_tendency(
        expected_net_surface_energy_flux, target_data["qf"], target_data["mld"]
    )
    timestep_seconds = TIMESTEP / datetime.timedelta(seconds=1)
    expected_sst = input_data["sst"] + timestep_seconds * expected_sst_tendency
    expected_output = {**fluxes, "sst": expected_sst}
    assert set(output_data) == set(expected_output)
    for name in output_data:
        torch.testing.assert_close(output_data[name], expected_output[name])


def test_mixed_layer_temperature_tendency():
    f_net = torch.tensor([10.0])
    q_flux = torch.tensor([5.0])
    depth = torch.tensor([100.0])
    result = mixed_layer_temperature_tendency(
        f_net, q_flux, depth, density=5.0, specific_heat=3.0
    )
    expected_result = (f_net + q_flux) / (5 * 3 * depth)
    torch.testing.assert_close(result, expected_result)


SURFACE_FLUX_NAMES = [
    "DLWRFsfc",
    "ULWRFsfc",
    "DSWRFsfc",
    "USWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
]


def _slab_ocean_config(
    surface_flux_source: Literal["prediction", "data"] = "prediction",
) -> OceanConfig:
    return OceanConfig(
        surface_temperature_name="sst",
        ocean_fraction_name="of",
        slab=SlabOceanConfig(
            mixed_layer_depth_name="mld",
            q_flux_name="qf",
            surface_flux_source=surface_flux_source,
        ),
    )


def test_ocean_slab_data_surface_flux():
    config = _slab_ocean_config(surface_flux_source="data")
    ocean = config.build(
        in_names=["sst"], out_names=["sst", *SURFACE_FLUX_NAMES], timestep=TIMESTEP
    )
    # data fluxes net to -4 W/m^2; the predicted fluxes would net to -400
    target_fluxes = {k: torch.tensor([2.0]) for k in SURFACE_FLUX_NAMES}
    gen_fluxes = {k: torch.tensor([200.0]) for k in SURFACE_FLUX_NAMES}
    target_data = {
        **target_fluxes,
        "mld": torch.tensor([25.0]),
        "of": torch.tensor([0.8]),
        "qf": torch.tensor([40.0]),
    }
    input_data = {"sst": torch.tensor([20.0])}
    gen_data = {**gen_fluxes, "sst": torch.tensor([25.0])}
    output_data = ocean(input_data, gen_data, target_data)
    # (F_net + Q) / (rho * c_p * h) * dt, by hand
    expected_sst = (
        torch.tensor([20.0])
        + (-4.0 + 40.0)
        / (DENSITY_OF_WATER * SPECIFIC_HEAT_OF_WATER * 25.0)
        * TIMESTEP.total_seconds()
    )
    torch.testing.assert_close(output_data["sst"], expected_sst)
    for name in SURFACE_FLUX_NAMES:
        torch.testing.assert_close(output_data[name], gen_fluxes[name])


@pytest.mark.parametrize(
    "surface_flux_source, expected_extra_names",
    [("prediction", []), ("data", SURFACE_FLUX_NAMES)],
)
def test_ocean_slab_target_names(surface_flux_source, expected_extra_names):
    config = _slab_ocean_config(surface_flux_source=surface_flux_source)
    out_names = ["sst", "foo", *SURFACE_FLUX_NAMES]
    # forcing names (which steps add to their input names) never get the
    # fluxes: that would make the predicted fluxes prognostic.
    assert sorted(config.forcing_names) == ["mld", "of", "qf"]
    assert sorted(config.target_names(out_names)) == sorted(
        ["mld", "of", "qf", *expected_extra_names]
    )


def test_ocean_target_names_resolves_flux_names_from_out_names():
    config = _slab_ocean_config(surface_flux_source="data")
    cam_flux_names = ["FLDS", "FLUS", "FSDS", "FSUS", "LHFLX", "SHFLX"]
    assert sorted(config.target_names(["sst", *cam_flux_names])) == sorted(
        ["mld", "of", "qf", *cam_flux_names]
    )


def test_ocean_slab_data_surface_flux_requires_flux_outputs():
    config = _slab_ocean_config(surface_flux_source="data")
    with pytest.raises(ValueError, match="sfc_down_lw_radiative_flux"):
        config.build(in_names=["sst"], out_names=["sst"], timestep=TIMESTEP)


def test_slab_ocean_config_invalid_surface_flux_source():
    with pytest.raises(ValueError, match="surface_flux_source"):
        SlabOceanConfig(
            mixed_layer_depth_name="mld",
            q_flux_name="qf",
            surface_flux_source="bad",  # type: ignore[arg-type]
        )


def test_ocean_config_without_surface_flux_source_loads_as_prediction():
    # configs serialized before surface_flux_source existed must load unchanged
    state = {
        "surface_temperature_name": "sst",
        "ocean_fraction_name": "of",
        "interpolate": False,
        "slab": {"mixed_layer_depth_name": "mld", "q_flux_name": "qf"},
    }
    config = dacite.from_dict(OceanConfig, state, config=dacite.Config(strict=True))
    assert config.slab is not None
    assert config.slab.surface_flux_source == "prediction"
    assert sorted(config.target_names(["sst", *SURFACE_FLUX_NAMES])) == [
        "mld",
        "of",
        "qf",
    ]
