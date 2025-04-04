import datetime
from collections import namedtuple
from typing import Iterable, List, Literal, Optional, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.aggregator import OneStepAggregator
from fme.ace.aggregator.plotting import plot_paneled_data
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.ocean import OceanConfig, SlabOceanConfig
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.typing_ import TensorDict
from fme.diffusion.loss import WeightedMappingLossConfig
from fme.diffusion.stepper import (
    DiffusionStepper,
    DiffusionStepperConfig,
    _combine_normalizers,
)

SphericalData = namedtuple("SphericalData", ["data", "area_weights", "vertical_coord"])
TIMESTEP = datetime.timedelta(hours=6)
DEVICE = fme.get_device()


def get_data(names: Iterable[str], n_samples, n_time) -> SphericalData:
    data_dict = {}
    n_lat, n_lon, nz = 5, 5, 7

    lats = torch.linspace(-89.5, 89.5, n_lat)  # arbitary choice
    for name in names:
        data_dict[name] = torch.rand(n_samples, n_time, n_lat, n_lon, device=DEVICE)
    area_weights = fme.spherical_area_weights(lats, n_lon).to(DEVICE)
    ak, bk = torch.arange(nz), torch.arange(nz)
    vertical_coord = HybridSigmaPressureCoordinate(ak, bk)
    data = BatchData.new_on_device(
        data=data_dict,
        time=xr.DataArray(
            np.zeros((n_samples, n_time)),
            dims=["sample", "time"],
        ),
    )
    return SphericalData(data, area_weights, vertical_coord)


def get_scalar_data(names, value):
    return {n: np.array([value], dtype=np.float32) for n in names}


@pytest.mark.parametrize(
    "in_names,out_names,ocean_config,expected_all_names",
    [
        (["a"], ["b"], None, ["a", "b"]),
        (["a"], ["a", "b"], None, ["a", "b"]),
        (["a", "b"], ["b"], None, ["a", "b"]),
        (["a", "b"], ["a", "b"], None, ["a", "b"]),
        (
            ["a", "b"],
            ["a", "b"],
            OceanConfig("a", "mask"),
            ["a", "b", "mask"],
        ),
        (
            ["a", "b"],
            ["a", "b"],
            OceanConfig("a", "b"),
            ["a", "b"],
        ),
        (
            ["a", "b"],
            ["a", "b"],
            OceanConfig("a", "of", False, SlabOceanConfig("c", "d")),
            ["a", "b", "of", "c", "d"],
        ),
    ],
)
def test_stepper_config_all_names_property(
    in_names, out_names, ocean_config, expected_all_names
):
    config = DiffusionStepperConfig(
        builder=MagicMock(),
        in_names=in_names,
        out_names=out_names,
        normalization=MagicMock(),
        ocean=ocean_config,
    )
    # check there are no duplications
    assert len(config.all_names) == len(set(config.all_names))
    # check the right items are in there using sets to ignore order
    assert set(config.all_names) == set(expected_all_names)


def test_train_on_batch_normalizer_changes_only_norm_data():
    torch.manual_seed(0)
    data = get_data(["a", "b"], n_samples=5, n_time=2).data
    area = torch.ones((5, 5), device=DEVICE)
    gridded_operations = LatLonOperations(area)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    normalization_config = NormalizationConfig(
        means=get_scalar_data(["a", "b"], 0.0),
        stds=get_scalar_data(["a", "b"], 1.0),
    )
    config = DiffusionStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": torch.nn.Identity()}),
        in_names=["a", "b"],
        out_names=["a", "b"],
        normalization=normalization_config,
        loss=WeightedMappingLossConfig(type="MSE"),
    )
    stepper = config.get_stepper(
        (5, 5), gridded_operations, vertical_coordinate, TIMESTEP
    )
    stepped = stepper.train_on_batch(data=data, optimization=NullOptimization())
    assert torch.allclose(
        stepped.gen_data["a"], stepped.normalize(stepped.gen_data)["a"]
    )  # as std=1, mean=0, no change
    normalization_config.stds = get_scalar_data(["a", "b"], 2.0)
    config.normalization = normalization_config
    config.loss_normalization = NormalizationConfig(
        means=get_scalar_data(["a", "b"], 0.0),
        stds=get_scalar_data(["a", "b"], 3.0),
    )
    stepper = config.get_stepper(
        (5, 5), gridded_operations, vertical_coordinate, TIMESTEP
    )
    stepped_double_std = stepper.train_on_batch(
        data=data, optimization=NullOptimization()
    )
    assert torch.allclose(
        stepped.gen_data["a"], stepped_double_std.gen_data["a"], rtol=1e-4
    )
    assert torch.allclose(
        stepped.gen_data["a"],
        2.0 * stepped_double_std.normalize(stepped_double_std.gen_data)["a"],
        rtol=1e-4,
    )
    assert torch.allclose(
        stepped.target_data["a"],
        2.0 * stepped_double_std.normalize(stepped_double_std.target_data)["a"],
        rtol=1e-4,
    )
    assert torch.allclose(
        stepped.metrics["loss"], 9.0 * stepped_double_std.metrics["loss"], rtol=1e-4
    )  # mse scales with std**2


def test_train_on_batch_addition_series():
    torch.manual_seed(0)

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    n_steps = 4
    data_with_ic: BatchData = get_data(["a", "b"], n_samples=5, n_time=n_steps + 1).data
    area = torch.ones((5, 5), device=DEVICE)
    gridded_operations = LatLonOperations(area)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    config = DiffusionStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": AddOne()}),
        in_names=["a", "b"],
        out_names=["a", "b"],
        normalization=NormalizationConfig(
            means=get_scalar_data(["a", "b"], 0.0),
            stds=get_scalar_data(["a", "b"], 1.0),
        ),
        loss=WeightedMappingLossConfig(type="MSE"),
    )
    stepper = config.get_stepper(
        (5, 5), gridded_operations, vertical_coordinate, TIMESTEP
    )
    stepped = stepper.train_on_batch(data=data_with_ic, optimization=NullOptimization())
    # output of train_on_batch does not include the initial condition
    assert stepped.gen_data["a"].shape == (5, 1, n_steps + 1, 5, 5)

    for i in range(n_steps - 1):
        assert torch.allclose(
            stepped.normalize(stepped.gen_data)["a"][:, :, i] + 1,
            stepped.normalize(stepped.gen_data)["a"][:, :, i + 1],
        )
        assert torch.allclose(
            stepped.normalize(stepped.gen_data)["b"][:, :, i] + 1,
            stepped.normalize(stepped.gen_data)["b"][:, :, i + 1],
        )
        assert torch.allclose(
            stepped.gen_data["a"][:, :, i] + 1, stepped.gen_data["a"][:, :, i + 1]
        )
        assert torch.allclose(
            stepped.gen_data["b"][:, :, i] + 1, stepped.gen_data["b"][:, :, i + 1]
        )
    assert torch.allclose(
        stepped.normalize(stepped.target_data)["a"],
        data_with_ic.data["a"][:, None],
    )
    assert torch.allclose(
        stepped.normalize(stepped.target_data)["b"],
        data_with_ic.data["b"][:, None],
    )


def test_train_on_batch_with_prescribed_ocean():
    torch.manual_seed(0)

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    n_steps = 3
    data: BatchData = get_data(["a", "b", "mask"], n_samples=5, n_time=n_steps + 1).data
    data.data["mask"][:] = 0
    data.data["mask"][:, :, :, 0] = 1
    stds = {
        "a": np.array([2.0], dtype=np.float32),
        "b": np.array([3.0], dtype=np.float32),
    }
    area = torch.ones((5, 5), device=DEVICE)
    gridded_operations = LatLonOperations(area)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    config = DiffusionStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": AddOne()}),
        in_names=["a", "b"],
        out_names=["a", "b"],
        normalization=NormalizationConfig(
            means=get_scalar_data(["a", "b"], 0.0),
            stds=stds,
        ),
        ocean=OceanConfig("b", "mask"),
    )
    stepper = config.get_stepper(
        area.shape, gridded_operations, vertical_coordinate, TIMESTEP
    )
    stepped = stepper.train_on_batch(data, optimization=NullOptimization())
    for i in range(n_steps - 1):
        # "a" should be increasing by 1 according to AddOne
        torch.testing.assert_close(
            stepped.normalize(stepped.gen_data)["a"][:, :, i] + 1,
            stepped.normalize(stepped.gen_data)["a"][:, :, i + 1],
        )
        # "b" should be increasing by 1 where the mask says don't prescribe
        # note the 1: selection for the last dimension in following two assertions
        torch.testing.assert_close(
            stepped.normalize(stepped.gen_data)["b"][:, :, i, :, 1:] + 1,
            stepped.normalize(stepped.gen_data)["b"][:, :, i + 1, :, 1:],
        )
        # now check that the 0th index in last dimension has been overwritten
        torch.testing.assert_close(
            stepped.normalize(stepped.gen_data)["b"][:, :, i, :, 0],
            stepped.normalize({"b": stepped.target_data["b"]})["b"][:, :, i, :, 0],
        )


def test_reloaded_stepper_gives_same_prediction():
    torch.manual_seed(0)
    config = DiffusionStepperConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
        ),
        in_names=["a", "b"],
        out_names=["a", "b"],
        normalization=NormalizationConfig(
            means={"a": 0.0, "b": 0.0},
            stds={"a": 1.0, "b": 1.0},
        ),
    )
    shapes = {
        "a": (1, 2, 5, 5),
        "b": (1, 2, 5, 5),
    }
    area = torch.ones((5, 5), device=DEVICE)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    stepper = config.get_stepper(
        img_shape=shapes["a"][-2:],
        gridded_operations=LatLonOperations(area),
        vertical_coordinate=vertical_coordinate,
        timestep=TIMESTEP,
    )
    area = torch.ones((5, 5), device=DEVICE)
    new_stepper = DiffusionStepper.from_state(stepper.get_state())
    data = get_data(["a", "b"], n_samples=5, n_time=2).data
    first_result = stepper.train_on_batch(
        data=data,
        optimization=NullOptimization(),
    )
    second_result = new_stepper.train_on_batch(
        data=data,
        optimization=NullOptimization(),
    )
    assert torch.allclose(first_result.metrics["loss"], second_result.metrics["loss"])
    assert torch.allclose(first_result.gen_data["a"], second_result.gen_data["a"])
    assert torch.allclose(first_result.gen_data["b"], second_result.gen_data["b"])
    assert torch.allclose(first_result.target_data["a"], second_result.target_data["a"])
    assert torch.allclose(first_result.target_data["b"], second_result.target_data["b"])


class ReturnZerosModule(torch.nn.Module):
    """
    Returns zeros with the correct number of out channels. Creates an unused
    parameter so that optimization has something to gnaw on.
    """

    def __init__(self, n_in_channels, n_out_channels) -> None:
        super().__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self._param = torch.nn.Parameter(
            torch.tensor(0.0, device=get_device())
        )  # unused

    def forward(self, x):
        assert torch.all(~torch.isnan(x))
        batch_size, n_channels, nlat, nlon = x.shape
        assert n_channels == self.n_in_channels
        zero = torch.zeros(
            batch_size, self.n_out_channels, nlat, nlon, device=get_device()
        )
        return zero + self._param


def _setup_and_train_on_batch(
    data: TensorDict,
    in_names,
    out_names,
    ocean_config: Optional[OceanConfig],
    optimization_config: Optional[OptimizationConfig],
):
    """Sets up the requisite classes to run train_on_batch."""
    module = ReturnZerosModule(len(in_names), len(out_names))

    if optimization_config is None:
        optimization: Union[NullOptimization, Optimization] = NullOptimization()
    else:
        optimization = optimization_config.build(modules=[module], max_epochs=2)

    area = torch.ones((5, 5), device=DEVICE)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    config = DiffusionStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": module}),
        in_names=in_names,
        out_names=out_names,
        normalization=NormalizationConfig(
            means=get_scalar_data(set(in_names + out_names), 0.0),
            stds=get_scalar_data(set(in_names + out_names), 1.0),
        ),
        ocean=ocean_config,
    )
    stepper = config.get_stepper(
        area.shape, LatLonOperations(area), vertical_coordinate, TIMESTEP
    )
    return stepper.train_on_batch(data, optimization=optimization)


@pytest.mark.parametrize(
    "is_input,is_output,is_prescribed",
    [
        pytest.param(True, True, True, id="in_out_prescribed"),
        pytest.param(True, True, False, id="in_out_not_prescribed"),
        pytest.param(False, True, False, id="out_only_not_prescribed"),
    ],
)
@pytest.mark.parametrize("n_forward_steps", [1, 2, 3], ids=lambda p: f"k={p}")
@pytest.mark.parametrize("is_train", [True, False], ids=["is_train", ""])
def test_train_on_batch(n_forward_steps, is_input, is_output, is_train, is_prescribed):
    in_names, out_names = ["a"], ["a"]
    if is_input:
        in_names.append("b")
    if is_output:
        out_names.append("b")
    all_names = sorted(list(set(in_names).union(set(out_names))))

    if is_prescribed:
        mask_name = "mask"
        all_names.append(mask_name)
        in_names.append(mask_name)
        ocean_config = OceanConfig("b", mask_name)
    else:
        ocean_config = None

    data, _, _ = get_data(all_names, 3, n_forward_steps + 1)

    if is_train:
        optimization = OptimizationConfig()
    else:
        optimization = None

    _setup_and_train_on_batch(data, in_names, out_names, ocean_config, optimization)


@pytest.mark.parametrize("n_forward_steps", [1, 2, 3])
def test_train_on_batch_one_step_aggregator(n_forward_steps):
    in_names, out_names, all_names = ["a"], ["a"], ["a"]
    data, _, _ = get_data(all_names, 3, n_forward_steps + 1)
    stepper = _get_stepper(in_names, out_names, ocean_config=None, module_name="AddOne")
    nx, ny = 5, 5
    lat_lon_coordinates = LatLonCoordinates(torch.arange(nx), torch.arange(ny))
    # keep area weights ones for simplicity
    lat_lon_coordinates._area_weights = torch.ones(nx, ny)
    aggregator = OneStepAggregator(lat_lon_coordinates, save_diagnostics=False)

    stepped = stepper.train_on_batch(data, optimization=NullOptimization())
    assert stepped.gen_data["a"].shape[2] == n_forward_steps + 1

    aggregator.record_batch(stepped)
    logs = aggregator.get_logs("one_step")

    gen = data.data["a"].select(dim=1, index=0) + 1
    tar = data.data["a"].select(dim=1, index=1)

    bias = torch.mean(gen - tar)
    assert np.isclose(bias.item(), logs["one_step/mean/weighted_bias/a"])

    residual_gen = torch.ones((5, 5))
    residual_tar = tar[0] - data.data["a"].select(dim=1, index=0)[0]
    residual_imgs = [[residual_gen.cpu().numpy()], [residual_tar.cpu().numpy()]]
    residual_plot = plot_paneled_data(residual_imgs, diverging=True)
    assert np.allclose(
        residual_plot.to_data_array(),
        logs["one_step/snapshot/image-residual/a"].to_data_array(),
    )

    full_field_gen = gen.mean(dim=0)
    full_field_tar = tar.mean(dim=0)
    full_field_plot = plot_paneled_data(
        [
            [full_field_gen.cpu().numpy()],
            [full_field_tar.cpu().numpy()],
        ],
        diverging=False,
    )
    assert np.allclose(
        full_field_plot.to_data_array(),
        logs["one_step/mean_map/image-full-field/a"].to_data_array(),
    )


class Multiply(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


def _get_stepper(
    in_names: List[str],
    out_names: List[str],
    ocean_config: Optional[OceanConfig] = None,
    module_name: Literal["AddOne", "ChannelSum", "RepeatChannel"] = "AddOne",
    **kwargs,
):
    if module_name == "AddOne":

        class AddOne(torch.nn.Module):
            def forward(self, x):
                return x + 1

        module_config = {"module": AddOne()}
    elif module_name == "ChannelSum":
        # convenient for testing stepper with more inputs than outputs
        class ChannelSum(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_input: Optional[torch.Tensor] = None

            def forward(self, x):
                self.last_input = x
                return x.sum(dim=-3, keepdim=True)

        module_config = {"module": ChannelSum()}
    elif module_name == "RepeatChannel":
        # convenient for testing stepper with more outputs than inputs
        class RepeatChannel(torch.nn.Module):
            def forward(self, x):
                return x.repeat(1, 2, 1, 1)

        module_config = {"module": RepeatChannel()}

    all_names = list(set(in_names + out_names))
    area = torch.ones((5, 5))
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    config = DiffusionStepperConfig(
        builder=ModuleSelector(type="prebuilt", config=module_config),
        in_names=in_names,
        out_names=out_names,
        normalization=NormalizationConfig(
            means={n: np.array([0.0], dtype=np.float32) for n in all_names},
            stds={n: np.array([1.0], dtype=np.float32) for n in all_names},
        ),
        ocean=ocean_config,
        **kwargs,
    )
    return config.get_stepper(
        (5, 5), LatLonOperations(area), vertical_coordinate, TIMESTEP
    )


def test_step():
    stepper = _get_stepper(["a", "b"], ["a", "b"])
    input_data = {x: torch.rand(3, 5, 5).to(DEVICE) for x in ["a", "b"]}

    output = stepper.step(input_data, {})

    torch.testing.assert_close(output["a"], input_data["a"] + 1)
    torch.testing.assert_close(output["b"], input_data["b"] + 1)


def test_step_with_diagnostic():
    stepper = _get_stepper(["a"], ["a", "c"], module_name="RepeatChannel")
    input_data = {"a": torch.rand(3, 5, 5).to(DEVICE)}
    output = stepper.step(input_data, {})
    torch.testing.assert_close(output["a"], input_data["a"])
    torch.testing.assert_close(output["c"], input_data["a"])


def test_step_with_forcing_and_diagnostic():
    stepper = _get_stepper(["a", "b"], ["a", "c"])
    input_data = {x: torch.rand(3, 5, 5).to(DEVICE) for x in ["a", "b"]}
    output = stepper.step(input_data, {})
    torch.testing.assert_close(output["a"], input_data["a"] + 1)
    assert "b" not in output
    assert "c" in output


def test_step_with_prescribed_ocean():
    stepper = _get_stepper(
        ["a", "b"], ["a", "b"], ocean_config=OceanConfig("a", "mask")
    )
    input_data = {x: torch.rand(3, 5, 5).to(DEVICE) for x in ["a", "b"]}
    ocean_data = {x: torch.rand(3, 5, 5).to(DEVICE) for x in ["a", "mask"]}
    output = stepper.step(input_data, ocean_data)
    expected_a_output = torch.where(
        torch.round(ocean_data["mask"]).to(int) == 1,
        ocean_data["a"],
        input_data["a"] + 1,
    )
    torch.testing.assert_close(output["a"], expected_a_output)
    torch.testing.assert_close(output["b"], input_data["b"] + 1)
    assert set(output) == {"a", "b"}


def get_data_for_predict(
    n_steps, forcing_names: List[str]
) -> Tuple[PrognosticState, BatchData]:
    n_samples = 3
    input_data = BatchData.new_on_device(
        data={"a": torch.rand(n_samples, 1, 5, 5).to(DEVICE)},
        time=xr.DataArray(
            np.zeros((n_samples, 1)),
            dims=["sample", "time"],
        ),
    ).get_start(
        prognostic_names=["a"],
        n_ic_timesteps=1,
    )
    forcing_data = BatchData.new_on_device(
        data={
            name: torch.rand(3, n_steps + 1, 5, 5).to(DEVICE) for name in forcing_names
        },
        time=xr.DataArray(
            np.zeros((n_samples, n_steps + 1)),
            dims=["sample", "time"],
        ),
    )
    return input_data, forcing_data


def test_predict():
    stepper = _get_stepper(["a"], ["a"])
    n_steps = 3
    input_data, forcing_data = get_data_for_predict(n_steps, forcing_names=[])
    forcing_data.data = {}
    output, new_input_data = stepper.predict(input_data, forcing_data)
    xr.testing.assert_allclose(forcing_data.time[:, 1:], output.time)
    variable = "a"
    assert output.data[variable].size(dim=1) == n_steps
    torch.testing.assert_close(
        output.data[variable][:, -1],
        input_data.as_batch_data().data[variable][:, 0] + n_steps,
    )
    assert isinstance(new_input_data, PrognosticState)
    new_input_state = new_input_data.as_batch_data()
    assert isinstance(new_input_state, BatchData)
    torch.testing.assert_close(
        new_input_state.data[variable][:, 0], output.data[variable][:, -1]
    )
    assert new_input_state.time.equals(output.time[:, -1:])


def test_predict_with_forcing():
    stepper = _get_stepper(["a", "b"], ["a"], module_name="ChannelSum")
    n_steps = 3
    input_data, forcing_data = get_data_for_predict(n_steps, forcing_names=["b"])
    output, new_input_data = stepper.predict(input_data, forcing_data)
    assert "b" not in output.data
    assert output.data["a"].size(dim=1) == n_steps
    xr.testing.assert_allclose(forcing_data.time[:, 1:], output.time)
    torch.testing.assert_close(
        output.data["a"][:, 0],
        input_data.as_batch_data().data["a"][:, 0] + forcing_data.data["b"][:, 0],
    )
    assert isinstance(new_input_data, PrognosticState)
    new_input_state = new_input_data.as_batch_data()
    assert isinstance(new_input_state, BatchData)
    torch.testing.assert_close(new_input_state.data["a"][:, 0], output.data["a"][:, -1])
    assert "b" not in new_input_state.data
    for n in range(1, n_steps):
        expected_a_output = output.data["a"][:, n - 1] + forcing_data.data["b"][:, n]
        torch.testing.assert_close(output.data["a"][:, n], expected_a_output)
    xr.testing.assert_equal(output.time, forcing_data.time[:, 1:])
    assert new_input_state.time.equals(output.time[:, -1:])


def test_predict_with_ocean():
    stepper = _get_stepper(["a"], ["a"], ocean_config=OceanConfig("a", "mask"))
    n_steps = 3
    input_data, forcing_data = get_data_for_predict(
        n_steps, forcing_names=["a", "mask"]
    )
    output, new_input_data = stepper.predict(input_data, forcing_data)
    xr.testing.assert_allclose(forcing_data.time[:, 1:], output.time)
    assert "mask" not in output.data
    assert output.data["a"].size(dim=1) == n_steps
    for n in range(n_steps):
        previous_a = (
            input_data.as_batch_data().data["a"][:, 0]
            if n == 0
            else output.data["a"][:, n - 1]
        )
        expected_a_output = torch.where(
            torch.round(forcing_data.data["mask"][:, n + 1]).to(int) == 1,
            forcing_data.data["a"][:, n + 1],
            previous_a + 1,
        )
        torch.testing.assert_close(output.data["a"][:, n], expected_a_output)
    assert isinstance(new_input_data, PrognosticState)
    new_input_state = new_input_data.as_batch_data()
    assert isinstance(new_input_state, BatchData)
    torch.testing.assert_close(new_input_state.data["a"][:, 0], output.data["a"][:, -1])
    assert new_input_state.time.equals(output.time[:, -1:])


def test_next_step_forcing_names():
    stepper = _get_stepper(
        ["a", "b", "c"],
        ["a"],
        module_name="ChannelSum",
        next_step_forcing_names=["c"],
    )
    input_data, forcing_data = get_data_for_predict(n_steps=1, forcing_names=["b", "c"])
    stepper.predict(input_data, forcing_data)
    torch.testing.assert_close(
        stepper.module.module.last_input[:, 1, :], forcing_data.data["b"][:, 0]
    )
    torch.testing.assert_close(
        stepper.module.module.last_input[:, 2, :], forcing_data.data["c"][:, 1]
    )


def test__combine_normalizers():
    vars = ["prog_0", "prog_1", "diag_0"]
    full_field_normalizer = StandardNormalizer(
        means={var: torch.rand(3) for var in vars},
        stds={var: torch.rand(3) for var in vars},
        fill_nans_on_normalize=True,
        fill_nans_on_denormalize=True,
    )
    residual_normalizer = StandardNormalizer(
        means={var: torch.rand(3) for var in ["prog_0", "prog_1"]},
        stds={var: torch.rand(3) for var in ["prog_0", "prog_1"]},
    )
    combined_normalizer = _combine_normalizers(
        residual_normalizer=residual_normalizer,
        model_normalizer=full_field_normalizer,
    )
    assert combined_normalizer.fill_nans_on_normalize
    assert combined_normalizer.fill_nans_on_denormalize
    for var in combined_normalizer.means:
        if "prog" in var:
            assert torch.allclose(
                combined_normalizer.means[var], residual_normalizer.means[var]
            )
            assert torch.allclose(
                combined_normalizer.stds[var], residual_normalizer.stds[var]
            )
        else:
            assert torch.allclose(
                combined_normalizer.means[var], full_field_normalizer.means[var]
            )
            assert torch.allclose(
                combined_normalizer.stds[var], full_field_normalizer.stds[var]
            )


def test_stepper_from_state_using_resnorm_has_correct_normalizer():
    # If originally configured with a residual normalizer, the
    # stepper loaded from state should have the appropriately combined
    # full field and residual values in its loss_normalizer
    torch.manual_seed(0)
    full_field_means = {"a": 0.0, "b": 0.0, "diagnostic": 0.0}
    full_field_stds = {"a": 1.0, "b": 1.0, "diagnostic": 1.0}
    # residual scalings might have diagnostic variables but the stepper
    # should detect which prognostic variables to use from the set
    residual_means = {"a": 1.0, "b": 1.0, "diagnostic": 1.0}
    residual_stds = {"a": 2.0, "b": 2.0, "diagnostic": 2.0}
    config = DiffusionStepperConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
        ),
        in_names=["a", "b"],
        out_names=["a", "b", "diagnostic"],
        normalization=NormalizationConfig(means=full_field_means, stds=full_field_stds),
        residual_normalization=NormalizationConfig(
            means=residual_means, stds=residual_stds
        ),
    )
    shapes = {
        "a": (1, 1, 5, 5),
        "b": (1, 1, 5, 5),
        "diagnostic": (1, 1, 5, 5),
    }
    area = torch.ones((5, 5), device=DEVICE)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    orig_stepper = config.get_stepper(
        img_shape=shapes["a"][-2:],
        gridded_operations=LatLonOperations(area),
        vertical_coordinate=vertical_coordinate,
        timestep=TIMESTEP,
    )
    stepper_from_state = DiffusionStepper.from_state(orig_stepper.get_state())

    for stepper in [orig_stepper, stepper_from_state]:
        assert stepper.loss_normalizer.means == {
            **residual_means,
            "diagnostic": full_field_means["diagnostic"],
        }
        assert stepper.loss_normalizer.stds == {
            **residual_stds,
            "diagnostic": full_field_stds["diagnostic"],
        }
        assert stepper.normalizer.means == full_field_means
        assert stepper.normalizer.stds == full_field_stds
