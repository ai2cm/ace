from datetime import timedelta

import numpy as np
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import BatchData
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import NullOptimization
from fme.core.registry.module import ModuleSelector
from fme.core.timing import GlobalTimer

from ..ace.stepper import SingleModuleStepperConfig
from .multi_call import MultiCallConfig, get_multi_call_name

TEST_CONFIG = MultiCallConfig(
    forcing_name="CO2",
    forcing_multipliers={"_quadrupled_co2": 4, "_halved_co2": 0.5},
    output_names=["OLR", "ASR", "bar_0", "bar_10"],
)


def _step(input, next_step_forcing, *_):
    names = TEST_CONFIG.output_names
    prediction = {k: input["CO2"].detach().clone() for k in names}
    return prediction


def test_multi_call_names():
    assert set(TEST_CONFIG.names) == {
        "OLR_quadrupled_co2",
        "ASR_quadrupled_co2",
        "bar_quadrupled_co2_0",
        "bar_quadrupled_co2_10",
        "OLR_halved_co2",
        "ASR_halved_co2",
        "bar_halved_co2_0",
        "bar_halved_co2_10",
    }


def test_multi_call():
    config = TEST_CONFIG
    multi_call = config.build(_step)
    co2_value = 2.0
    shape = (1, 2, 3, 3)
    initial_condition = {"temperature": torch.ones(shape)}
    co2_data = {"CO2": torch.full(shape, co2_value)}

    output = multi_call.step(initial_condition | co2_data, {})

    assert set(output) == set(config.names)
    for name in config.output_names:
        for multiplier_name, multiplier_value in config.forcing_multipliers.items():
            # the _step method in this test module returns predictions equal
            # to the CO2 value in the forcing data, hence this check.
            multi_call_name = get_multi_call_name(name, multiplier_name)
            torch.testing.assert_close(
                output[multi_call_name],
                co2_data["CO2"] * multiplier_value,
            )
    torch.testing.assert_close(co2_data["CO2"], torch.full(shape, co2_value))


def get_scalar_data(names, value):
    return {n: value for n in names}


def _get_stepper_config(
    in_names, out_names, all_names, multi_call_config, include_loss
):
    class CustomModule(torch.nn.Module):
        def forward(self, x):
            channel_dim = -3
            input_co2 = x.select(channel_dim, 0)
            input_temperature = x.select(channel_dim, 1)
            new_temperature = input_temperature + input_co2
            output_olr = input_co2 * input_temperature
            return torch.stack([new_temperature, output_olr], dim=channel_dim)

        def eval(self):
            pass

    config = SingleModuleStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": CustomModule()}),
        in_names=in_names,
        out_names=out_names,
        normalization=NormalizationConfig(
            means=get_scalar_data(all_names, 0.0),
            stds=get_scalar_data(all_names, 1.0),
        ),
        loss=WeightedMappingLossConfig(type="MSE", weights={"temperature": 1.0}),
        multi_call=multi_call_config,
        include_multi_call_in_loss=include_loss,
    )

    return config


def test_integration_with_stepper():
    img_shape = (5, 5)
    full_shape = (1, 3, *img_shape)
    horizontal_coord = LatLonCoordinates(
        torch.zeros(img_shape[0], device=fme.get_device()),
        torch.zeros(img_shape, device=fme.get_device()),
    )
    vertical_coord = HybridSigmaPressureCoordinate(
        torch.tensor([1.0], device=fme.get_device()),
        torch.tensor([0.0], device=fme.get_device()),
    )
    timestep = timedelta(seconds=1)

    in_names = ["CO2", "temperature"]
    out_names = ["temperature", "OLR"]
    multi_call_names = ["OLR_doubled_co2"]
    multi_call_config = MultiCallConfig(
        forcing_name="CO2",
        forcing_multipliers={"_doubled_co2": 2},
        output_names=["OLR"],
    )
    expected_all_names = set(in_names + out_names + multi_call_names)
    config = _get_stepper_config(
        in_names, out_names, expected_all_names, multi_call_config, True
    )

    assert set(config.diagnostic_names) == set(out_names).difference(in_names).union(
        multi_call_names
    )
    assert set(config.all_names) == expected_all_names
    stepper = config.get_stepper(
        dataset_info=DatasetInfo(
            horizontal_coordinates=horizontal_coord,
            vertical_coordinate=vertical_coord,
            timestep=timestep,
        ),
    )
    time = xr.DataArray([[1, 1, 1]], dims=["sample", "time"])
    data = BatchData(
        {
            n: torch.ones(full_shape, device=fme.get_device())
            for n in expected_all_names
        },
        time,
    )
    with GlobalTimer():
        output = stepper.train_on_batch(data, NullOptimization())
    assert "OLR_doubled_co2" in output.gen_data
    assert "temperature" in output.gen_data
    assert "OLR" in output.gen_data
    assert "OLR_doubled_co2" in output.target_data
    expected_loss = output.metrics["loss_step_0"] + output.metrics["loss_step_1"]
    torch.testing.assert_close(output.metrics["loss"], expected_loss)
    # this value check is based on the implementation of the CustomModule above
    # first output time step
    co2 = data.data["CO2"][0, 0].cpu().numpy()  # assuming constant in time
    for t in range(3):
        output_temperature = output.gen_data["temperature"][0, 0, t].cpu().numpy()
        np.testing.assert_allclose(output_temperature, co2 + t)
        if t > 0:
            # only check diagnostic OLR for the output time steps, not initial condition
            input_temperature = (
                output.gen_data["temperature"][0, 0, t - 1].cpu().numpy()
            )
            olr = output.gen_data["OLR"][0, 0, t].cpu().numpy()
            np.testing.assert_allclose(olr, input_temperature * co2)
            olr_doubled = output.gen_data["OLR_doubled_co2"][0, 0, t].cpu().numpy()
            # note for the double-called case, it should use the temperature from the
            # non-doubled CO2 case
            np.testing.assert_allclose(olr_doubled, input_temperature * 2 * co2)

    # rerun without including multi-call in loss
    multi_call_config = MultiCallConfig(
        forcing_name="CO2",
        forcing_multipliers={"_doubled_co2": 2},
        output_names=["OLR"],
    )
    config = _get_stepper_config(
        in_names, out_names, expected_all_names, multi_call_config, False
    )
    stepper = config.get_stepper(
        dataset_info=DatasetInfo(
            horizontal_coordinates=horizontal_coord,
            vertical_coordinate=vertical_coord,
            timestep=timestep,
        ),
    )
    with GlobalTimer():
        output_without_loss = stepper.train_on_batch(data, NullOptimization())

    torch.testing.assert_close(
        output_without_loss.metrics["loss"],
        output_without_loss.metrics["loss_step_0"]
        + output_without_loss.metrics["loss_step_1"],
    )
