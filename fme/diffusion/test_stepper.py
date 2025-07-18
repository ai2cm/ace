import datetime
from collections import namedtuple
from collections.abc import Iterable
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.ocean import OceanConfig, SlabOceanConfig
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.typing_ import TensorDict
from fme.diffusion.loss import WeightedMappingLossConfig
from fme.diffusion.registry import ModuleSelector
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


class PassThrough(torch.nn.Module):
    def __init__(self, n_out_channels: int):
        super().__init__()
        self.n_out_channels = n_out_channels

    def forward(self, x, emb):
        return x[..., : self.n_out_channels, :, :]


@pytest.mark.parametrize("n_steps", [1])
def test_train_on_batch_addition_series(n_steps: int):
    torch.manual_seed(0)

    data_with_ic: BatchData = get_data(["a", "b"], n_samples=5, n_time=n_steps + 1).data
    area = torch.ones((5, 5), device=DEVICE)
    gridded_operations = LatLonOperations(area)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    config = DiffusionStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": PassThrough(2)}),
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

    n_steps = 1
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
        builder=ModuleSelector(type="prebuilt", config={"module": PassThrough(2)}),
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
    assert isinstance(stepped.gen_data["a"], torch.Tensor)
    assert isinstance(stepped.gen_data["b"], torch.Tensor)
    assert isinstance(stepped.target_data["a"], torch.Tensor)
    assert isinstance(stepped.target_data["b"], torch.Tensor)


def test_reloaded_stepper_gives_different_prediction():
    torch.manual_seed(0)
    config = DiffusionStepperConfig(
        builder=ModuleSelector(type="ConditionalSFNO", config={"scale_factor": 1}),
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
    # it's a stochastic model, so we expect different predictions
    assert not torch.allclose(
        first_result.metrics["loss"], second_result.metrics["loss"]
    )
    assert not torch.allclose(first_result.gen_data["a"], second_result.gen_data["a"])
    assert not torch.allclose(first_result.gen_data["b"], second_result.gen_data["b"])
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

    def forward(self, x, emb):
        assert torch.all(~torch.isnan(x))
        batch_size, n_channels, nlat, nlon = x.shape
        assert n_channels == self.n_in_channels + self.n_out_channels
        zero = torch.zeros(
            batch_size, self.n_out_channels, nlat, nlon, device=get_device()
        )
        return zero + self._param


def _setup_and_train_on_batch(
    data: TensorDict,
    in_names,
    out_names,
    ocean_config: OceanConfig | None,
    optimization_config: OptimizationConfig | None,
):
    """Sets up the requisite classes to run train_on_batch."""
    module = ReturnZerosModule(len(in_names), len(out_names))

    if optimization_config is None:
        optimization: NullOptimization | Optimization = NullOptimization()
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
@pytest.mark.parametrize("is_train", [True, False], ids=["is_train", ""])
def test_train_on_batch(is_input, is_output, is_train, is_prescribed):
    n_forward_steps = 1
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


class Multiply(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


def _get_stepper(
    in_names: list[str],
    out_names: list[str],
    ocean_config: OceanConfig | None = None,
    module_name: Literal["PassThrough"] = "PassThrough",
    **kwargs,
):
    if module_name == "PassThrough":
        module_config = {"module": PassThrough(len(out_names))}
    else:
        raise ValueError(f"Unknown module name: {module_name}")

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
    assert isinstance(output["a"], torch.Tensor)
    assert isinstance(output["b"], torch.Tensor)


def test_step_with_diagnostic():
    stepper = _get_stepper(["a"], ["a", "c"], module_name="PassThrough")
    input_data = {"a": torch.rand(3, 5, 5).to(DEVICE)}
    output = stepper.step(input_data, {})
    assert isinstance(output["a"], torch.Tensor)
    assert isinstance(output["c"], torch.Tensor)


def test_step_with_forcing_and_diagnostic():
    stepper = _get_stepper(["a", "b"], ["a", "c"])
    input_data = {x: torch.rand(3, 5, 5).to(DEVICE) for x in ["a", "b"]}
    output = stepper.step(input_data, {})
    assert isinstance(output["a"], torch.Tensor)
    assert "b" not in output
    assert "c" in output


def test_step_with_prescribed_ocean():
    stepper = _get_stepper(
        ["a", "b"], ["a", "b"], ocean_config=OceanConfig("a", "mask")
    )
    input_data = {x: torch.rand(3, 5, 5).to(DEVICE) for x in ["a", "b"]}
    ocean_data = {x: torch.rand(3, 5, 5).to(DEVICE) for x in ["a", "mask"]}
    output = stepper.step(input_data, ocean_data)
    assert isinstance(output["a"], torch.Tensor)
    assert isinstance(output["b"], torch.Tensor)
    assert set(output) == {"a", "b"}


def get_data_for_predict(
    n_steps, forcing_names: list[str]
) -> tuple[PrognosticState, BatchData]:
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
    assert isinstance(output.data[variable], torch.Tensor)
    assert isinstance(new_input_data, PrognosticState)
    new_input_state = new_input_data.as_batch_data()
    assert isinstance(new_input_state, BatchData)
    assert new_input_state.time.equals(output.time[:, -1:])


def test_predict_with_forcing():
    stepper = _get_stepper(["a", "b"], ["a"], module_name="PassThrough")
    n_steps = 3
    input_data, forcing_data = get_data_for_predict(n_steps, forcing_names=["b"])
    output, new_input_data = stepper.predict(input_data, forcing_data)
    assert "b" not in output.data
    assert output.data["a"].size(dim=1) == n_steps
    xr.testing.assert_allclose(forcing_data.time[:, 1:], output.time)
    assert isinstance(output.data["a"], torch.Tensor)
    assert isinstance(new_input_data, PrognosticState)
    new_input_state = new_input_data.as_batch_data()
    assert isinstance(new_input_state, BatchData)
    assert "b" not in new_input_state.data
    for n in range(1, n_steps):
        expected_a_output = output.data["a"][:, n - 1] + forcing_data.data["b"][:, n]
        assert isinstance(expected_a_output, torch.Tensor)
        assert output.data["a"][:, n].shape == expected_a_output.shape
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
        assert isinstance(expected_a_output, torch.Tensor)
        assert output.data["a"][:, n].shape == expected_a_output.shape
    assert isinstance(new_input_data, PrognosticState)
    new_input_state = new_input_data.as_batch_data()
    assert isinstance(new_input_state, BatchData)
    assert new_input_state.time.equals(output.time[:, -1:])


def test_next_step_forcing_names():
    stepper = _get_stepper(
        ["a", "b", "c"],
        ["a"],
        module_name="PassThrough",
        next_step_forcing_names=["c"],
    )
    input_data, forcing_data = get_data_for_predict(n_steps=1, forcing_names=["b", "c"])
    stepper.predict(input_data, forcing_data)


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
            torch.testing.assert_close(
                combined_normalizer.means[var], residual_normalizer.means[var]
            )
            torch.testing.assert_close(
                combined_normalizer.stds[var], residual_normalizer.stds[var]
            )
        else:
            torch.testing.assert_close(
                combined_normalizer.means[var], full_field_normalizer.means[var]
            )
            torch.testing.assert_close(
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
        builder=ModuleSelector(type="ConditionalSFNO", config={"scale_factor": 1}),
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
