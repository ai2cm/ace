import datetime
from collections import namedtuple
from typing import Iterable, List, Literal, Optional, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import fme
from fme.ace.inference.derived_variables import compute_stepped_derived_quantities
from fme.core import ClimateData, metrics
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.device import get_device
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.ocean import OceanConfig, SlabOceanConfig
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.registry import ModuleSelector
from fme.core.stepper import (
    CorrectorConfig,
    SingleModuleStepper,
    SingleModuleStepperConfig,
    SteppedData,
    _combine_normalizers,
)
from fme.core.typing_ import TensorDict

SphericalData = namedtuple("SphericalData", ["data", "area_weights", "sigma_coords"])
TIMESTEP = datetime.timedelta(hours=6)


def get_data(names: Iterable[str], n_samples, n_time) -> SphericalData:
    data = {}
    n_lat, n_lon, nz = 5, 5, 7

    lats = torch.linspace(-89.5, 89.5, n_lat)  # arbitary choice
    for name in names:
        data[name] = torch.rand(
            n_samples, n_time, n_lat, n_lon, device=fme.get_device()
        )
    area_weights = fme.spherical_area_weights(lats, n_lon).to(fme.get_device())
    ak, bk = torch.arange(nz), torch.arange(nz)
    sigma_coords = SigmaCoordinates(ak, bk)
    return SphericalData(data, area_weights, sigma_coords)


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
    config = SingleModuleStepperConfig(
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


def test_run_on_batch_normalizer_changes_only_norm_data():
    torch.manual_seed(0)
    data = get_data(["a", "b"], n_samples=5, n_time=2).data
    area = torch.ones((5, 5), device=fme.get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": torch.nn.Identity()}),
        in_names=["a", "b"],
        out_names=["a", "b"],
        normalization=NormalizationConfig(
            means=get_scalar_data(["a", "b"], 0.0),
            stds=get_scalar_data(["a", "b"], 1.0),
        ),
        loss=WeightedMappingLossConfig(type="MSE"),
    )
    stepper = config.get_stepper((5, 5), area, sigma_coordinates, TIMESTEP)
    stepped = stepper.run_on_batch(data=data, optimization=MagicMock())
    assert torch.allclose(
        stepped.gen_data["a"], stepped.gen_data_norm["a"]
    )  # as std=1, mean=0, no change
    config.normalization.stds = get_scalar_data(["a", "b"], 2.0)
    config.loss_normalization = NormalizationConfig(
        means=get_scalar_data(["a", "b"], 0.0),
        stds=get_scalar_data(["a", "b"], 3.0),
    )
    stepper = config.get_stepper((5, 5), area, sigma_coordinates, TIMESTEP)
    stepped_double_std = stepper.run_on_batch(data=data, optimization=MagicMock())
    assert torch.allclose(
        stepped.gen_data["a"], stepped_double_std.gen_data["a"], rtol=1e-4
    )
    assert torch.allclose(
        stepped.gen_data["a"], 2.0 * stepped_double_std.gen_data_norm["a"], rtol=1e-4
    )
    assert torch.allclose(
        stepped.target_data["a"],
        2.0 * stepped_double_std.target_data_norm["a"],
        rtol=1e-4,
    )
    assert torch.allclose(
        stepped.metrics["loss"], 9.0 * stepped_double_std.metrics["loss"], rtol=1e-4
    )  # mse scales with std**2


def test_run_on_batch_addition_series():
    torch.manual_seed(0)

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    n_steps = 4
    data_with_ic = get_data(["a", "b"], n_samples=5, n_time=n_steps + 1).data
    area = torch.ones((5, 5), device=fme.get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": AddOne()}),
        in_names=["a", "b"],
        out_names=["a", "b"],
        normalization=NormalizationConfig(
            means=get_scalar_data(["a", "b"], 0.0),
            stds=get_scalar_data(["a", "b"], 1.0),
        ),
        loss=WeightedMappingLossConfig(type="MSE"),
    )
    stepper = config.get_stepper((5, 5), area, sigma_coordinates, TIMESTEP)
    stepped = stepper.run_on_batch(
        data=data_with_ic, optimization=MagicMock(), n_forward_steps=n_steps
    )
    # output of run_on_batch does not include the initial condition
    assert stepped.gen_data["a"].shape == (5, n_steps, 5, 5)
    data = {k: data_with_ic[k][:, 1:] for k in data_with_ic}

    for i in range(n_steps - 1):
        assert torch.allclose(
            stepped.gen_data_norm["a"][:, i] + 1, stepped.gen_data_norm["a"][:, i + 1]
        )
        assert torch.allclose(
            stepped.gen_data_norm["b"][:, i] + 1, stepped.gen_data_norm["b"][:, i + 1]
        )
        assert torch.allclose(
            stepped.gen_data["a"][:, i] + 1, stepped.gen_data["a"][:, i + 1]
        )
        assert torch.allclose(
            stepped.gen_data["b"][:, i] + 1, stepped.gen_data["b"][:, i + 1]
        )
    assert torch.allclose(stepped.target_data_norm["a"], data["a"])
    assert torch.allclose(stepped.target_data_norm["b"], data["b"])


def test_run_on_batch_with_prescribed_ocean():
    torch.manual_seed(0)

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    n_steps = 3
    data = get_data(["a", "b", "mask"], n_samples=5, n_time=n_steps + 1).data
    data["mask"] = torch.zeros_like(data["mask"], dtype=torch.int)
    data["mask"][:, :, :, 0] = 1
    stds = {
        "a": np.array([2.0], dtype=np.float32),
        "b": np.array([3.0], dtype=np.float32),
        "mask": np.array([1.0], dtype=np.float32),
    }
    area = torch.ones((5, 5), device=fme.get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": AddOne()}),
        in_names=["a", "b"],
        out_names=["a", "b"],
        normalization=NormalizationConfig(
            means=get_scalar_data(["a", "b", "mask"], 0.0),
            stds=stds,
        ),
        ocean=OceanConfig("b", "mask"),
    )
    stepper = config.get_stepper(area.shape, area, sigma_coordinates, TIMESTEP)
    stepped = stepper.run_on_batch(
        data, optimization=MagicMock(), n_forward_steps=n_steps
    )
    for i in range(n_steps - 1):
        # "a" should be increasing by 1 according to AddOne
        torch.testing.assert_close(
            stepped.gen_data_norm["a"][:, i] + 1, stepped.gen_data_norm["a"][:, i + 1]
        )
        # "b" should be increasing by 1 where the mask says don't prescribe
        # note the 1: selection for the last dimension in following two assertions
        torch.testing.assert_close(
            stepped.gen_data_norm["b"][:, i, :, 1:] + 1,
            stepped.gen_data_norm["b"][:, i + 1, :, 1:],
        )
        # now check that the 0th index in last dimension has been overwritten
        torch.testing.assert_close(
            stepped.gen_data_norm["b"][:, i, :, 0],
            stepped.target_data_norm["b"][:, i, :, 0],
        )


def test_reloaded_stepper_gives_same_prediction():
    torch.manual_seed(0)
    config = SingleModuleStepperConfig(
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
        "a": (1, 1, 5, 5),
        "b": (1, 1, 5, 5),
    }
    area = torch.ones((5, 5), device=fme.get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    stepper = config.get_stepper(
        img_shape=shapes["a"][-2:],
        area=area,
        sigma_coordinates=sigma_coordinates,
        timestep=TIMESTEP,
    )
    area = torch.ones((5, 5), device=fme.get_device())
    new_stepper = SingleModuleStepper.from_state(
        stepper.get_state(), area=area, sigma_coordinates=sigma_coordinates
    )
    data = get_data(["a", "b"], n_samples=5, n_time=2).data
    first_result = stepper.run_on_batch(
        data=data,
        optimization=NullOptimization(),
        n_forward_steps=1,
    )
    second_result = new_stepper.run_on_batch(
        data=data,
        optimization=NullOptimization(),
        n_forward_steps=1,
    )
    assert torch.allclose(first_result.metrics["loss"], second_result.metrics["loss"])
    assert torch.allclose(first_result.gen_data["a"], second_result.gen_data["a"])
    assert torch.allclose(first_result.gen_data["b"], second_result.gen_data["b"])
    assert torch.allclose(
        first_result.gen_data_norm["a"], second_result.gen_data_norm["a"]
    )
    assert torch.allclose(
        first_result.gen_data_norm["b"], second_result.gen_data_norm["b"]
    )
    assert torch.allclose(
        first_result.target_data_norm["a"], second_result.target_data_norm["a"]
    )
    assert torch.allclose(
        first_result.target_data_norm["b"], second_result.target_data_norm["b"]
    )
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


def _setup_and_run_on_batch(
    data: TensorDict,
    in_names,
    out_names,
    ocean_config: Optional[OceanConfig],
    n_forward_steps,
    optimization_config: Optional[OptimizationConfig],
):
    """Sets up the requisite classes to run run_on_batch."""
    module = ReturnZerosModule(len(in_names), len(out_names))

    if optimization_config is None:
        optimization: Union[NullOptimization, Optimization] = NullOptimization()
    else:
        optimization = optimization_config.build(module.parameters(), 2)

    area = torch.ones((5, 5), device=fme.get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": module}),
        in_names=in_names,
        out_names=out_names,
        normalization=NormalizationConfig(
            means=get_scalar_data(set(in_names + out_names), 0.0),
            stds=get_scalar_data(set(in_names + out_names), 1.0),
        ),
        ocean=ocean_config,
    )
    stepper = config.get_stepper(area.shape, area, sigma_coordinates, TIMESTEP)
    return stepper.run_on_batch(
        data, optimization=optimization, n_forward_steps=n_forward_steps
    )


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
def test_run_on_batch(n_forward_steps, is_input, is_output, is_train, is_prescribed):
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

    data, area_weights, sigma_coords = get_data(all_names, 3, n_forward_steps + 1)

    if is_train:
        optimization = OptimizationConfig()
    else:
        optimization = None

    _setup_and_run_on_batch(
        data, in_names, out_names, ocean_config, n_forward_steps, optimization
    )


class Multiply(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


@pytest.mark.parametrize(
    "global_only, terms_to_modify, force_positive",
    [
        (True, "none", False),
        (True, "precipitation", False),
        (True, "evaporation", False),
        (False, "advection_and_precipitation", False),
        (False, "advection_and_evaporation", False),
        (False, "advection_and_precipitation", True),
    ],
)
def test_stepper_corrector(global_only: bool, terms_to_modify, force_positive: bool):
    torch.random.manual_seed(0)
    n_forward_steps = 5
    device = get_device()
    data = {
        "PRESsfc": 10.0 + torch.rand(size=(3, n_forward_steps + 1, 5, 5)).to(device),
        "specific_total_water_0": -0.2
        + torch.rand(size=(3, n_forward_steps + 1, 5, 5)).to(device),
        "specific_total_water_1": torch.rand(size=(3, n_forward_steps + 1, 5, 5)).to(
            device
        ),
        "PRATEsfc": torch.rand(size=(3, n_forward_steps + 1, 5, 5)).to(device),
        "LHTFLsfc": torch.rand(size=(3, n_forward_steps + 1, 5, 5)).to(device),
        "tendency_of_total_water_path_due_to_advection": torch.rand(
            size=(3, n_forward_steps + 1, 5, 5)
        ).to(device),
    }
    sigma_coordinates = SigmaCoordinates(
        ak=torch.asarray([3.0, 1.0, 0.0]), bk=torch.asarray([0.0, 0.6, 1.0])
    ).to(device)
    area_weights = 1.0 + torch.rand(size=(5, 5)).to(device)

    if force_positive:
        force_positive_names = ["specific_total_water_0"]
    else:
        force_positive_names = []

    corrector_config = CorrectorConfig(
        conserve_dry_air=True,
        zero_global_mean_moisture_advection=True,
        moisture_budget_correction=terms_to_modify,
        force_positive_names=force_positive_names,
    )

    mean_advection = metrics.weighted_mean(
        data["tendency_of_total_water_path_due_to_advection"],
        weights=area_weights,
        dim=[-2, -1],
    )
    assert (mean_advection.abs() > 0.0).all()

    # use a randomly initialized Linear layer for the module
    # using PrebuiltBuilder
    stepper_config = SingleModuleStepperConfig(
        builder=ModuleSelector(
            type="prebuilt",
            config={
                "module": Multiply(1.5).to(device),
            },
        ),
        in_names=list(data.keys()),
        out_names=list(data.keys()),
        normalization=NormalizationConfig(
            means={key: 0.0 for key in data.keys()},
            stds={key: 1.0 for key in data.keys()},
        ),
        corrector=corrector_config,
    )
    stepper = stepper_config.get_stepper(
        img_shape=data["PRESsfc"].shape[2:],
        area=area_weights,
        sigma_coordinates=sigma_coordinates,
        timestep=TIMESTEP,
    )
    # run the stepper on the data
    with torch.no_grad():
        stepped = stepper.run_on_batch(
            data=data,
            optimization=NullOptimization(),
            n_forward_steps=n_forward_steps,
        )

    stepped = compute_stepped_derived_quantities(
        stepped, sigma_coordinates=sigma_coordinates, timestep=TIMESTEP
    )

    # check that the budget residual is zero
    budget_residual = stepped.gen_data["total_water_path_budget_residual"]
    if global_only:
        budget_residual = metrics.weighted_mean(
            budget_residual, weights=area_weights, dim=[-2, -1]
        )
    budget_residual = budget_residual.cpu().numpy()
    if terms_to_modify != "none":
        if global_only:
            mean_axis: Tuple[int, ...] = (0,)
        else:
            mean_axis = (0, 2, 3)
        # first assert on timeseries, easier to look at
        np.testing.assert_almost_equal(
            np.abs(budget_residual).mean(axis=mean_axis), 0.0, decimal=6
        )
        np.testing.assert_almost_equal(budget_residual, 0.0, decimal=5)

    # check there is no mean advection
    mean_advection = (
        metrics.weighted_mean(
            stepped.gen_data["tendency_of_total_water_path_due_to_advection"],
            weights=area_weights,
            dim=[-2, -1],
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_almost_equal(mean_advection[:, 1:], 0.0, decimal=6)

    # check that the dry air is conserved
    dry_air = (
        metrics.weighted_mean(
            ClimateData(stepped.gen_data).surface_pressure_due_to_dry_air(
                sigma_coordinates
            ),
            weights=area_weights,
            dim=[-2, -1],
        )
        .cpu()
        .numpy()
    )
    dry_air_nonconservation = np.abs(dry_air[:, 1:] - dry_air[:, :-1])
    np.testing.assert_almost_equal(dry_air_nonconservation, 0.0, decimal=3)

    # check that positive forcing is enforced
    if force_positive:
        for name in force_positive_names:
            assert stepped.gen_data[name].min() >= 0.0


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
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    config = SingleModuleStepperConfig(
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
    return config.get_stepper((5, 5), area, sigma_coordinates, TIMESTEP)


def test_step():
    stepper = _get_stepper(["a", "b"], ["a", "b"])
    input_data = {x: torch.rand(3, 5, 5) for x in ["a", "b"]}

    output = stepper.step(input_data, {})

    torch.testing.assert_close(output["a"], input_data["a"] + 1)
    torch.testing.assert_close(output["b"], input_data["b"] + 1)


def test_step_with_diagnostic():
    stepper = _get_stepper(["a"], ["a", "c"], module_name="RepeatChannel")
    input_data = {"a": torch.rand(3, 5, 5)}
    output = stepper.step(input_data, {})
    torch.testing.assert_close(output["a"], input_data["a"])
    torch.testing.assert_close(output["c"], input_data["a"])


def test_step_with_forcing_and_diagnostic():
    stepper = _get_stepper(["a", "b"], ["a", "c"])
    input_data = {x: torch.rand(3, 5, 5) for x in ["a", "b"]}
    output = stepper.step(input_data, {})
    torch.testing.assert_close(output["a"], input_data["a"] + 1)
    assert "b" not in output
    assert "c" in output


def test_step_with_prescribed_ocean():
    stepper = _get_stepper(
        ["a", "b"], ["a", "b"], ocean_config=OceanConfig("a", "mask")
    )
    input_data = {x: torch.rand(3, 5, 5) for x in ["a", "b", "mask"]}
    ocean_data = {x: torch.rand(3, 5, 5) for x in ["a", "mask"]}
    output = stepper.step(input_data, ocean_data)
    expected_a_output = torch.where(
        torch.round(ocean_data["mask"]).to(int) == 1,
        ocean_data["a"],
        input_data["a"] + 1,
    )
    torch.testing.assert_close(output["a"], expected_a_output)
    torch.testing.assert_close(output["b"], input_data["b"] + 1)
    assert set(output) == {"a", "b"}


def test_predict():
    stepper = _get_stepper(["a", "b"], ["a", "b"])
    n_steps = 3
    input_data = {x: torch.rand(3, 5, 5) for x in ["a", "b"]}
    forcing_data = {}
    output = stepper.predict(input_data, forcing_data, n_steps)
    for variable in ["a", "b"]:
        assert output[variable].size(dim=1) == n_steps
        torch.testing.assert_close(
            output[variable][:, -1], input_data[variable] + n_steps
        )


def test_predict_with_forcing():
    stepper = _get_stepper(["a", "b"], ["a"], module_name="ChannelSum")
    n_steps = 3
    input_data = {"a": torch.rand(3, 5, 5)}
    forcing_data = {"b": torch.rand(3, n_steps + 1, 5, 5)}
    output = stepper.predict(input_data, forcing_data, n_steps)
    assert "b" not in output
    assert output["a"].size(dim=1) == n_steps
    torch.testing.assert_close(
        output["a"][:, 0], input_data["a"] + forcing_data["b"][:, 0]
    )
    for n in range(1, n_steps):
        expected_a_output = output["a"][:, n - 1] + forcing_data["b"][:, n]
        torch.testing.assert_close(output["a"][:, n], expected_a_output)


def test_predict_with_ocean():
    stepper = _get_stepper(["a"], ["a"], ocean_config=OceanConfig("a", "mask"))
    n_steps = 3
    input_data = {"a": torch.rand(3, 5, 5)}
    forcing_data = {x: torch.rand(3, n_steps + 1, 5, 5) for x in ["a", "mask"]}
    output = stepper.predict(input_data, forcing_data, n_steps)
    assert "mask" not in output
    assert output["a"].size(dim=1) == n_steps
    for n in range(n_steps):
        previous_a = input_data["a"] if n == 0 else output["a"][:, n - 1]
        expected_a_output = torch.where(
            torch.round(forcing_data["mask"][:, n + 1]).to(int) == 1,
            forcing_data["a"][:, n + 1],
            previous_a + 1,
        )
        torch.testing.assert_close(output["a"][:, n], expected_a_output)


def test_next_step_forcing_names():
    stepper = _get_stepper(
        ["a", "b", "c"],
        ["a"],
        module_name="ChannelSum",
        next_step_forcing_names=["c"],
    )
    input_data = {x: torch.rand(1, 5, 5) for x in ["a"]}
    forcing_data = {x: torch.rand(1, 2, 5, 5) for x in ["b", "c"]}
    stepper.predict(input_data, forcing_data, 1)
    torch.testing.assert_close(
        stepper.module.module.last_input[:, 1, :], forcing_data["b"][:, 0]
    )
    torch.testing.assert_close(
        stepper.module.module.last_input[:, 2, :], forcing_data["c"][:, 1]
    )


def test_prepend_initial_condition():
    nt = 3
    x = torch.rand(3, nt, 5).to(fme.get_device())
    x_normed = (x - x.mean()) / x.std()
    stepped = SteppedData(
        gen_data={"a": x, "b": x + 1},
        gen_data_norm={"a": x_normed, "b": x_normed + 1},
        target_data={"a": x, "b": x + 1},
        target_data_norm={"a": x_normed, "b": x_normed + 1},
        metrics={"loss": torch.tensor(0.0)},
    )
    ic = {
        "a": torch.rand(3, 5).to(fme.get_device()),
        "b": torch.rand(3, 5).to(fme.get_device()),
    }
    ic_normed = {k: (v - v.mean()) / v.std() for k, v in ic.items()}
    prepended = stepped.prepend_initial_condition(ic, ic_normed)
    for v in ["a", "b"]:
        assert torch.allclose(prepended.gen_data[v][:, 0], ic[v])
        assert torch.allclose(prepended.gen_data_norm[v][:, 0], ic_normed[v])
        assert torch.allclose(prepended.target_data[v][:, 0], ic[v])
        assert torch.allclose(prepended.target_data_norm[v][:, 0], ic_normed[v])


def test__combine_normalizers():
    vars = ["prog_0", "prog_1", "diag_0"]
    full_field_normalizer = StandardNormalizer(
        means={var: torch.rand(3) for var in vars},
        stds={var: torch.rand(3) for var in vars},
    )
    residual_normalizer = StandardNormalizer(
        means={var: torch.rand(3) for var in ["prog_0", "prog_1"]},
        stds={var: torch.rand(3) for var in ["prog_0", "prog_1"]},
    )
    combined_normalizer = _combine_normalizers(
        residual_normalizer=residual_normalizer, model_normalizer=full_field_normalizer
    )
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
    full_field_normalization = {
        "means": {"a": 0.0, "b": 0.0, "diagnostic": 0.0},
        "stds": {"a": 1.0, "b": 1.0, "diagnostic": 1.0},
    }
    # residual scalings might have diagnostic variables but the stepper
    # should detect which prognostic variables to use from the set
    residual_normalization = {
        "means": {"a": 1.0, "b": 1.0, "diagnostic": 1.0},
        "stds": {"a": 2.0, "b": 2.0, "diagnostic": 2.0},
    }
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
        ),
        in_names=["a", "b"],
        out_names=["a", "b", "diagnostic"],
        normalization=NormalizationConfig(**full_field_normalization),
        residual_normalization=NormalizationConfig(**residual_normalization),
    )
    shapes = {
        "a": (1, 1, 5, 5),
        "b": (1, 1, 5, 5),
        "diagnostic": (1, 1, 5, 5),
    }
    area = torch.ones((5, 5), device=fme.get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    orig_stepper = config.get_stepper(
        img_shape=shapes["a"][-2:],
        area=area,
        sigma_coordinates=sigma_coordinates,
        timestep=TIMESTEP,
    )
    stepper_from_state = SingleModuleStepper.from_state(
        orig_stepper.get_state(), area=area, sigma_coordinates=sigma_coordinates
    )

    for stepper in [orig_stepper, stepper_from_state]:
        assert stepper.loss_normalizer.means == {"a": 1.0, "b": 1.0, "diagnostic": 0.0}
        assert stepper.loss_normalizer.stds == {"a": 2.0, "b": 2.0, "diagnostic": 1.0}
        assert stepper.normalizer.means == full_field_normalization["means"]
        assert stepper.normalizer.stds == full_field_normalization["stds"]


def test_stepper_effective_loss_scaling():
    custom_loss_weights = {"b": 2.0}
    loss_norm_means = {"a": 0.0, "b": 0.0}
    loss_norm_stds = {"a": 4.0, "b": 0.5}
    stepper = _get_stepper(
        in_names=["a", "b"],
        out_names=["a", "b"],
        loss=WeightedMappingLossConfig(weights=custom_loss_weights),
        loss_normalization=NormalizationConfig(
            means=loss_norm_means, stds=loss_norm_stds
        ),
    )
    assert stepper.effective_loss_scaling == {
        "a": torch.tensor(4.0),
        "b": torch.tensor(0.25),
    }
