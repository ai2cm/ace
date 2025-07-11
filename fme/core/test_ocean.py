import datetime

import torch

from fme.core.ocean import (
    Ocean,
    OceanConfig,
    SlabOceanConfig,
    mixed_layer_temperature_tendency,
)

TIMESTEP = datetime.timedelta(hours=6)


def test_ocean_prescribed():
    config = OceanConfig(surface_temperature_name="sst", ocean_fraction_name="of")
    ocean = Ocean(config, timestep=TIMESTEP)
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
    ocean = Ocean(config, timestep=TIMESTEP)
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
