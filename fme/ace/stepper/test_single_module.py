import dataclasses
import datetime
import os
import pathlib
from collections import namedtuple
from collections.abc import Iterable, Mapping
from typing import Literal
from unittest.mock import patch

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.aggregator import OneStepAggregator
from fme.ace.aggregator.plotting import plot_paneled_data
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.inference.test_evaluator import (
    save_plus_one_stepper,
    validate_stepper_config,
    validate_stepper_multi_call,
    validate_stepper_ocean,
)
from fme.ace.registry.sfno import SphericalFourierNeuralOperatorBuilder
from fme.ace.stepper.single_module import (
    AtmosphereCorrectorConfig,
    Stepper,
    StepperConfig,
    StepperOverrideConfig,
    TrainOutput,
    get_serialized_stepper_vertical_coordinate,
    load_stepper,
    load_stepper_config,
)
from fme.ace.testing import DimSizes
from fme.core import AtmosphereData
from fme.core.coordinates import (
    DimSize,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
    VerticalCoordinate,
)
from fme.core.dataset_info import DatasetInfo, MissingDatasetInfo
from fme.core.device import get_device
from fme.core.generics.optimization import OptimizationABC
from fme.core.loss import WeightedMappingLossConfig
from fme.core.mask_provider import MaskProvider
from fme.core.masking import StaticMaskingConfig
from fme.core.multi_call import MultiCallConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.ocean import OceanConfig
from fme.core.optimization import (
    CheckpointConfig,
    NullOptimization,
    Optimization,
    OptimizationConfig,
)
from fme.core.registry.module import ModuleSelector
from fme.core.step import SingleModuleStepConfig, StepSelector
from fme.core.testing.regression import validate_tensor_dict
from fme.core.training_history import TrainingJob
from fme.core.typing_ import EnsembleTensorDict

DIR = os.path.abspath(os.path.dirname(__file__))

SphericalData = namedtuple("SphericalData", ["data", "area_weights", "vertical_coord"])
TIMESTEP = datetime.timedelta(hours=6)
DEVICE = fme.get_device()
OCEAN_CONFIG = OceanConfig(surface_temperature_name="a", ocean_fraction_name="b")
MULTI_CALL_CONFIG = MultiCallConfig(
    forcing_name="co2", forcing_multipliers={"_b": 0.5}, output_names=["ULWRFtoa"]
)
LOAD_STEPPER_TESTS = {
    "override-ocean": (None, None, OCEAN_CONFIG, "keep", OCEAN_CONFIG, None),
    "persist-ocean": (OCEAN_CONFIG, None, "keep", "keep", OCEAN_CONFIG, None),
    "override-multi-call": (
        None,
        None,
        "keep",
        MULTI_CALL_CONFIG,
        None,
        MULTI_CALL_CONFIG,
    ),
    "persist-multi-call": (
        None,
        MULTI_CALL_CONFIG,
        "keep",
        "keep",
        None,
        MULTI_CALL_CONFIG,
    ),
    "override-all": (
        None,
        None,
        OCEAN_CONFIG,
        MULTI_CALL_CONFIG,
        OCEAN_CONFIG,
        MULTI_CALL_CONFIG,
    ),
    "persist-all": (
        OCEAN_CONFIG,
        MULTI_CALL_CONFIG,
        "keep",
        "keep",
        OCEAN_CONFIG,
        MULTI_CALL_CONFIG,
    ),
}


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


def get_dataset_info(
    img_shape=(5, 5),
    mask_provider=None,
    vertical_coordinate=None,
    horizontal_coordinate=None,
) -> DatasetInfo:
    if horizontal_coordinate is None:
        horizontal_coordinate = LatLonCoordinates(
            lat=torch.zeros(img_shape[-2]),
            lon=torch.zeros(img_shape[-1]),
        )
    if vertical_coordinate is None:
        vertical_coordinate = HybridSigmaPressureCoordinate(
            ak=torch.arange(7), bk=torch.arange(7)
        )
    return DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=TIMESTEP,
        mask_provider=mask_provider,
    )


def get_scalar_data(names, value):
    return {n: float(value) for n in names}


def test_train_on_batch_normalizer_changes_only_norm_data():
    torch.manual_seed(0)
    data = get_data(["a", "b"], n_samples=5, n_time=2).data
    normalization_config = NormalizationConfig(
        means=get_scalar_data(["a", "b"], 0.0),
        stds=get_scalar_data(["a", "b"], 1.0),
    )

    def get_stepper_config(normalization_config: NetworkAndLossNormalizationConfig):
        return StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="prebuilt", config={"module": torch.nn.Identity()}
                        ),
                        in_names=["a", "b"],
                        out_names=["a", "b"],
                        normalization=normalization_config,
                    )
                ),
            ),
            loss=WeightedMappingLossConfig(type="MSE"),
        )

    config = get_stepper_config(
        NetworkAndLossNormalizationConfig(network=normalization_config)
    )
    dataset_info = get_dataset_info()
    stepper = config.get_stepper(dataset_info)
    stepped = stepper.train_on_batch(data=data, optimization=NullOptimization())
    assert torch.allclose(
        stepped.gen_data["a"], stepped.normalize(stepped.gen_data)["a"]
    )  # as std=1, mean=0, no change
    config = get_stepper_config(
        NetworkAndLossNormalizationConfig(
            network=NormalizationConfig(
                means=get_scalar_data(["a", "b"], 0.0),
                stds=get_scalar_data(["a", "b"], 2.0),
            ),
            loss=NormalizationConfig(
                means=get_scalar_data(["a", "b"], 0.0),
                stds=get_scalar_data(["a", "b"], 3.0),
            ),
        )
    )
    stepper = config.get_stepper(dataset_info)
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
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt", config={"module": AddOne()}
                    ),
                    in_names=["a", "b"],
                    out_names=["a", "b"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means=get_scalar_data(["a", "b"], 0.0),
                            stds=get_scalar_data(["a", "b"], 1.0),
                        ),
                    ),
                )
            ),
        ),
        loss=WeightedMappingLossConfig(type="MSE"),
    )
    dataset_info = get_dataset_info()
    stepper = config.get_stepper(dataset_info)
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


def test_train_on_batch_crps_loss():
    torch.manual_seed(0)

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    n_steps = 4
    data_with_ic: BatchData = get_data(["a", "b"], n_samples=5, n_time=n_steps + 1).data

    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt", config={"module": AddOne()}
                    ),
                    in_names=["a", "b"],
                    out_names=["a", "b"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means=get_scalar_data(["a", "b"], 0.0),
                            stds=get_scalar_data(["a", "b"], 1.0),
                        ),
                    ),
                )
            ),
        ),
        n_ensemble=2,
        loss=WeightedMappingLossConfig(
            type="EnsembleLoss",
            kwargs={
                "crps_weight": 0.1,
                "energy_score_weight": 0.9,
            },
        ),
    )
    dataset_info = get_dataset_info()
    stepper = config.get_stepper(dataset_info)
    stepped = stepper.train_on_batch(data=data_with_ic, optimization=NullOptimization())
    # output of train_on_batch does not include the initial condition
    assert stepped.gen_data["a"].shape == (5, 2, n_steps + 1, 5, 5)


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
        "a": 2.0,
        "b": 3.0,
    }
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt", config={"module": AddOne()}
                    ),
                    in_names=["a", "b"],
                    out_names=["a", "b"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means=get_scalar_data(["a", "b"], 0.0),
                            stds=stds,
                        ),
                    ),
                    ocean=OceanConfig("b", "mask"),
                )
            ),
        ),
    )
    dataset_info = get_dataset_info()
    stepper = config.get_stepper(dataset_info)
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

    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="SphericalFourierNeuralOperatorNet",
                        config={"scale_factor": 1},
                    ),
                    in_names=["a", "b"],
                    out_names=["a", "b"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means=get_scalar_data(["a", "b"], 0.0),
                            stds=get_scalar_data(["a", "b"], 1.0),
                        ),
                    ),
                )
            ),
        ),
        loss=WeightedMappingLossConfig(type="MSE"),
    )
    dataset_info = get_dataset_info()
    stepper = config.get_stepper(dataset_info)
    new_stepper = Stepper.from_state(stepper.get_state())
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


def test_reloaded_stepper_has_metadata():
    stepper = _get_stepper(["a", "b"], ["a", "c"])
    # set the metadata manually for testing
    training_job_metadata = TrainingJob(
        git_sha="1234567890abcdef",
        job_id="some_run_id",
    )
    stepper.training_history.append(training_job_metadata)
    stepper_state = stepper.get_state()
    new_stepper = Stepper.from_state(stepper_state)
    assert new_stepper.training_history == stepper.training_history


def test_stepper_update_training_history():
    stepper = _get_stepper(["a"], ["b"], module_name="Linear")
    assert len(stepper.training_history) == 0

    # first update
    git_sha = "1234567890abcdef"
    job_id = "some_run_id_456"
    training_job = TrainingJob(git_sha=git_sha, job_id=job_id)
    stepper.update_training_history(training_job)
    assert len(stepper.training_history) == 1
    assert stepper.training_history[0].git_sha == git_sha
    assert stepper.training_history[0].job_id == job_id

    # second update with changed stepper module weights
    git_sha_2 = "9876543210fedcba"
    job_id_2 = "some_run_id_789"
    training_job_2 = TrainingJob(git_sha=git_sha_2, job_id=job_id_2)
    stepper._step_obj.modules[0].module.linear.weight.data.add_(0.01)
    stepper.update_training_history(training_job_2)
    assert len(stepper.training_history) == 2
    for i, job in enumerate([training_job, training_job_2]):
        assert stepper.training_history[i].git_sha == job.git_sha
        assert stepper.training_history[i].job_id == job.job_id


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
    data: BatchData,
    in_names,
    out_names,
    ocean_config: OceanConfig | None,
    optimization_config: OptimizationConfig | None,
    stepper_config_kwargs,
):
    """Sets up the requisite classes to run train_on_batch."""
    module = ReturnZerosModule(len(in_names), len(out_names))

    if optimization_config is None:
        optimization: NullOptimization | Optimization = NullOptimization()
    else:
        optimization = optimization_config.build(modules=[module], max_epochs=2)

    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(type="prebuilt", config={"module": module}),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means=get_scalar_data(set(in_names + out_names), 0.0),
                            stds=get_scalar_data(set(in_names + out_names), 1.0),
                        ),
                    ),
                    ocean=ocean_config,
                    **stepper_config_kwargs,
                )
            ),
        ),
        loss=WeightedMappingLossConfig(type="MSE"),
    )

    dataset_info = get_dataset_info()
    stepper = config.get_stepper(dataset_info)
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
@pytest.mark.parametrize(
    "with_activation_checkpointing", [True, False], ids=["act_ckpt", ""]
)
def test_train_on_batch(
    n_forward_steps,
    is_input,
    is_output,
    is_train,
    is_prescribed,
    with_activation_checkpointing,
):
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
        if with_activation_checkpointing:
            optimization = OptimizationConfig(
                checkpoint=CheckpointConfig(after_n_forward_steps=n_forward_steps - 1)
            )
        else:
            optimization = OptimizationConfig()
    else:
        optimization = None

    with patch("torch.utils.checkpoint.checkpoint") as mock_checkpoint:
        # have the mock call the module and return the step
        mock_checkpoint.side_effect = lambda f, x, **_: f(x)
        _setup_and_train_on_batch(
            data, in_names, out_names, ocean_config, optimization, {}
        )

        if is_train and with_activation_checkpointing:
            # should be called exactly once, for the final forward step
            mock_checkpoint.assert_called_once()
        else:
            mock_checkpoint.assert_not_called()


@pytest.mark.parametrize("n_forward_steps", [1, 2, 3])
def test_train_on_batch_one_step_aggregator(n_forward_steps):
    in_names, out_names, all_names = ["a"], ["a"], ["a"]
    data, _, _ = get_data(all_names, 3, n_forward_steps + 1)
    nx, ny = 5, 5
    stepper = _get_stepper(in_names, out_names, ocean_config=None, module_name="AddOne")
    lat_lon_coordinates = LatLonCoordinates(torch.arange(nx), torch.arange(ny))
    # keep area weights ones for simplicity
    lat_lon_coordinates._area_weights = torch.ones(nx, ny)
    ds_info = DatasetInfo(horizontal_coordinates=lat_lon_coordinates)
    aggregator = OneStepAggregator(ds_info, save_diagnostics=False)

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


@pytest.mark.parametrize(
    "global_only, terms_to_modify, force_positive",
    [
        (True, None, False),
        (True, "precipitation", False),
        (True, "evaporation", False),
        (False, "advection_and_precipitation", False),
        (False, "advection_and_evaporation", False),
        (False, "advection_and_precipitation", True),
    ],
)
@pytest.mark.parametrize("compute_derived_in_train_on_batch", [False, True])
def test_stepper_corrector(
    global_only: bool,
    terms_to_modify,
    force_positive: bool,
    compute_derived_in_train_on_batch: bool,
):
    torch.random.manual_seed(0)
    n_forward_steps = 5
    device = get_device()
    data = {
        "PRESsfc": 10.0 + torch.rand(size=(3, n_forward_steps + 1, 5, 5)),
        "specific_total_water_0": -0.2
        + torch.rand(size=(3, n_forward_steps + 1, 5, 5)),
        "specific_total_water_1": torch.rand(size=(3, n_forward_steps + 1, 5, 5)),
        "PRATEsfc": torch.rand(size=(3, n_forward_steps + 1, 5, 5)),
        "LHTFLsfc": torch.rand(size=(3, n_forward_steps + 1, 5, 5)),
        "tendency_of_total_water_path_due_to_advection": torch.rand(
            size=(3, n_forward_steps + 1, 5, 5)
        ),
    }
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.asarray([3.0, 1.0, 0.0]), bk=torch.asarray([0.0, 0.6, 1.0])
    ).to(device)
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.linspace(-89.5, 89.5, 5, device=device),
        lon=torch.linspace(-179.5, 179.5, 5, device=device),
    )
    dataset_info = get_dataset_info(
        vertical_coordinate=vertical_coordinate,
        horizontal_coordinate=horizontal_coordinate,
    )
    gridded_ops = dataset_info.gridded_operations

    if force_positive:
        force_positive_names = ["specific_total_water_0"]
    else:
        force_positive_names = []

    corrector_config = AtmosphereCorrectorConfig(
        conserve_dry_air=True,
        zero_global_mean_moisture_advection=True,
        moisture_budget_correction=terms_to_modify,
        force_positive_names=force_positive_names,
    )

    mean_advection = gridded_ops.area_weighted_mean(
        data["tendency_of_total_water_path_due_to_advection"].to(device)
    )
    assert (mean_advection.abs() > 0.0).all()

    stepper_config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt",
                        config={
                            "module": Multiply(1.5).to(device),
                        },
                    ),
                    in_names=list(data.keys()),
                    out_names=list(data.keys()),
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={key: 0.0 for key in data.keys()},
                            stds={key: 1.0 for key in data.keys()},
                        ),
                    ),
                    corrector=corrector_config,
                )
            ),
        ),
    )
    stepper = stepper_config.get_stepper(dataset_info)
    time = xr.DataArray(
        [
            [
                cftime.DatetimeProlepticGregorian(
                    2000, 1, int(i * 6 // 24) + 1, i * 6 % 24
                )
                for i in range(n_forward_steps + 1)
            ]
            for _ in range(3)
        ],
        dims=["sample", "time"],
    )
    batch_data = BatchData.new_on_cpu(
        data=data,
        time=time,
    ).to_device()
    # run the stepper on the data
    with torch.no_grad():
        stepped = stepper.train_on_batch(
            data=batch_data,
            optimization=NullOptimization(),
            compute_derived_variables=compute_derived_in_train_on_batch,
        )
    if not compute_derived_in_train_on_batch:
        stepped = stepped.compute_derived_variables()
    # check that the budget residual is zero
    budget_residual = stepped.gen_data["total_water_path_budget_residual"]
    if global_only:
        budget_residual = gridded_ops.area_weighted_mean(budget_residual)
    budget_residual = budget_residual.cpu().numpy()
    if terms_to_modify is not None:
        if global_only:
            mean_axis: tuple[int, ...] = (0,)
        else:
            mean_axis = (0, 2, 3)
        # first assert on timeseries, easier to look at
        np.testing.assert_almost_equal(
            np.abs(budget_residual).mean(axis=mean_axis), 0.0, decimal=6
        )
        np.testing.assert_almost_equal(budget_residual, 0.0, decimal=5)

    # check there is no mean advection
    mean_advection = (
        gridded_ops.area_weighted_mean(
            stepped.gen_data["tendency_of_total_water_path_due_to_advection"]
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_almost_equal(mean_advection[:, 1:], 0.0, decimal=6)

    # check that the dry air is conserved
    dry_air = (
        gridded_ops.area_weighted_mean(
            AtmosphereData(
                stepped.gen_data, vertical_coordinate
            ).surface_pressure_due_to_dry_air
        )
        .cpu()
        .numpy()
    )
    dry_air_nonconservation = np.abs(dry_air[:, 1:] - dry_air[:, :-1])
    np.testing.assert_almost_equal(dry_air_nonconservation, 0.0, decimal=3)

    # check that positive forcing is enforced
    if force_positive:
        for name in force_positive_names:
            assert stepped.gen_data[name][:, :, 1:].min() >= 0.0


def _get_stepper(
    in_names: list[str],
    out_names: list[str],
    ocean_config: OceanConfig | None = None,
    module_name: Literal["AddOne", "ChannelSum", "RepeatChannel", "Linear"] = "AddOne",
    norm_mean: float = 0.0,
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
                self.last_input: torch.Tensor | None = None

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
    elif module_name == "Linear":
        # convenient for testing a stepper with parameters
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x)

        module_config = {"module": Linear()}

    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(type="prebuilt", config=module_config),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means=get_scalar_data(set(in_names + out_names), norm_mean),
                            stds=get_scalar_data(set(in_names + out_names), 1.0),
                        ),
                    ),
                    ocean=ocean_config,
                    **kwargs,
                )
            ),
        ),
        loss=WeightedMappingLossConfig(type="MSE"),
    )
    dataset_info = get_dataset_info()
    return config.get_stepper(dataset_info)


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


@pytest.mark.parametrize("residual_prediction", [False, True])
def test_step_with_forcing_and_diagnostic(residual_prediction):
    norm_mean = 2.0
    stepper = _get_stepper(
        ["a", "b"],
        ["a", "c"],
        norm_mean=norm_mean,
        residual_prediction=residual_prediction,
    )
    input_data = {x: torch.rand(3, 5, 5).to(DEVICE) for x in ["a", "b"]}
    output = stepper.step(input_data, {})
    if residual_prediction:
        expected_a_output = 2 * input_data["a"] + 1 - norm_mean
    else:
        expected_a_output = input_data["a"] + 1
    torch.testing.assert_close(output["a"], expected_a_output)
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
    assert len(stepper.modules) == 1
    torch.testing.assert_close(
        stepper.modules[0].module.last_input[:, 1, :],
        forcing_data.data["b"][:, 0],
    )
    torch.testing.assert_close(
        stepper.modules[0].module.last_input[:, 2, :],
        forcing_data.data["c"][:, 1],
    )


def test_prepend_initial_condition():
    nt = 3
    batch_size = 3
    n_ensemble = 2
    x = torch.rand(3, n_ensemble, nt, 5).to(DEVICE)

    def normalize(x):
        result = {k: (v - 1) / 2 for k, v in x.items()}
        return result

    stepped = TrainOutput(
        gen_data=EnsembleTensorDict({"a": x, "b": x + 1}),
        target_data=EnsembleTensorDict({"a": x[:, :1] + 2, "b": x[:, :1] + 3}),
        time=xr.DataArray(np.zeros((batch_size, nt)), dims=["sample", "time"]),
        metrics={"loss": torch.tensor(0.0)},
        normalize=normalize,
    )
    ic_data = {
        # no sample dim in initial condition
        "a": torch.rand(batch_size, 1, 5).to(DEVICE),
        "b": torch.rand(batch_size, 1, 5).to(DEVICE),
    }
    ic = BatchData.new_on_device(
        data=ic_data,
        time=xr.DataArray(np.zeros((3, 1)), dims=["sample", "time"]),
    ).get_start(
        prognostic_names=["a", "b"],
        n_ic_timesteps=1,
    )
    prepended = stepped.prepend_initial_condition(ic)
    for v in ["a", "b"]:
        assert torch.allclose(prepended.gen_data[v][:, :, :1], ic_data[v][:, None, ...])
        assert torch.allclose(
            prepended.target_data[v][:, :, :1], ic_data[v][:, None, ...]
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
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="SphericalFourierNeuralOperatorNet",
                        config={"scale_factor": 1},
                    ),
                    in_names=["a", "b"],
                    out_names=["a", "b", "diagnostic"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means=full_field_means, stds=full_field_stds
                        ),
                        residual=NormalizationConfig(
                            means=residual_means, stds=residual_stds
                        ),
                    ),
                )
            ),
        )
    )
    orig_stepper = config.get_stepper(dataset_info=get_dataset_info())
    stepper_from_state = Stepper.from_state(orig_stepper.get_state())

    for stepper in [orig_stepper, stepper_from_state]:
        assert stepper.loss_obj.normalizer.means == {
            **residual_means,
            "diagnostic": full_field_means["diagnostic"],
        }
        assert stepper.loss_obj.normalizer.stds == {
            **residual_stds,
            "diagnostic": full_field_stds["diagnostic"],
        }
        assert stepper.normalizer.means == full_field_means
        assert stepper.normalizer.stds == full_field_stds


@pytest.mark.parametrize(
    (
        "serialized_ocean_config",
        "serialized_multi_call_config",
        "overriding_ocean_config",
        "overriding_multi_call_config",
        "expected_ocean_config",
        "expected_multi_call_config",
    ),
    list(LOAD_STEPPER_TESTS.values()),
    ids=list(LOAD_STEPPER_TESTS.keys()),
)
def test_load_stepper_and_load_stepper_config(
    tmp_path: pathlib.Path,
    serialized_ocean_config: OceanConfig | None,
    serialized_multi_call_config: MultiCallConfig | None,
    overriding_ocean_config: Literal["keep"] | OceanConfig | None,
    overriding_multi_call_config: Literal["keep"] | MultiCallConfig | None,
    expected_ocean_config: OceanConfig | None,
    expected_multi_call_config: MultiCallConfig | None,
):
    in_names = ["co2", "var", "a", "b"]
    fluxes = ["ULWRFtoa"]
    out_names = ["var", "a"] + fluxes
    stepper_path = tmp_path / "stepper"
    n_forward_steps = 8

    horizontal = [DimSize("grid_yt", 4), DimSize("grid_xt", 8)]
    dim_sizes = DimSizes(
        n_time=n_forward_steps + 1,
        horizontal=horizontal,
        nz_interface=4,
    )
    if serialized_multi_call_config is not None:
        multi_call_names = serialized_multi_call_config.names
    else:
        multi_call_names = []
    if (
        overriding_multi_call_config is not None
        and overriding_multi_call_config != "keep"
    ):
        multi_call_names += overriding_multi_call_config.names
    save_plus_one_stepper(
        stepper_path,
        in_names,
        out_names,
        normalization_names=set(in_names + out_names + multi_call_names),
        mean=0.0,
        std=1.0,
        data_shape=dim_sizes.shape_nd,
        ocean=serialized_ocean_config,
        multi_call=serialized_multi_call_config,
    )

    # First check that load_stepper_config and load_stepper functions load
    # the unmodified stepper when no StepperOverrideConfig is passed.
    stepper_config = load_stepper_config(stepper_path)
    validate_stepper_config(
        stepper_config, serialized_ocean_config, serialized_multi_call_config
    )

    stepper = load_stepper(stepper_path)
    validate_stepper_ocean(stepper, serialized_ocean_config)
    validate_stepper_multi_call(stepper, serialized_multi_call_config)

    # Then check that they load the appropriately modified config and
    # stepper when provided with a StepperOverrideConfig.
    stepper_override = StepperOverrideConfig(
        ocean=overriding_ocean_config, multi_call=overriding_multi_call_config
    )

    stepper_config = load_stepper_config(stepper_path, stepper_override)
    validate_stepper_config(
        stepper_config, expected_ocean_config, expected_multi_call_config
    )

    stepper = load_stepper(stepper_path, stepper_override)
    validate_stepper_ocean(stepper, expected_ocean_config)
    validate_stepper_multi_call(stepper, expected_multi_call_config)


def get_regression_stepper_and_data(
    crps_training: bool = False,
) -> tuple[Stepper, BatchData]:
    in_names = ["a", "b"]
    out_names = ["b", "c"]
    n_forward_steps = 2
    n_samples = 3
    img_shape = (9, 18)
    device = get_device()

    all_names = list(set(in_names + out_names))

    if crps_training:
        loss = WeightedMappingLossConfig(
            type="EnsembleLoss",
            kwargs={"crps_weight": 1.0, "energy_score_weight": 0.0},
        )
        n_ensemble: int = 2
    else:
        loss = WeightedMappingLossConfig(type="MSE")
        n_ensemble = 1

    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="SphericalFourierNeuralOperatorNet",
                        config=dataclasses.asdict(
                            SphericalFourierNeuralOperatorBuilder(
                                embed_dim=16,
                                num_layers=2,
                            )
                        ),
                    ),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={n: 0.1 for n in all_names},
                            stds={n: 1.1 for n in all_names},
                        ),
                    ),
                    ocean=None,
                )
            ),
        ),
        loss=loss,
        n_ensemble=n_ensemble,
    )

    dataset_info = get_dataset_info(img_shape=img_shape)
    stepper = config.get_stepper(dataset_info)
    data = BatchData(
        data={
            "a": torch.randn(n_samples, n_forward_steps + 1, *img_shape).to(device),
            "b": torch.randn(n_samples, n_forward_steps + 1, *img_shape).to(device),
            "c": torch.randn(n_samples, n_forward_steps + 1, *img_shape).to(device),
        },
        time=xr.DataArray(
            np.zeros((n_samples, n_forward_steps + 1)),
            dims=["sample", "time"],
        ),
    )
    return stepper, data


@pytest.mark.parametrize(
    "use_optimization",
    [
        pytest.param(True, id="optimization"),
        pytest.param(False, id="no_optimization"),
    ],
)
@pytest.mark.parametrize(
    "crps_training",
    [
        pytest.param(True, id="crps_training"),
        pytest.param(False, id="no_crps_training"),
    ],
)
def test_stepper_train_on_batch_regression(use_optimization: bool, crps_training: bool):
    torch.manual_seed(0)
    stepper, data = get_regression_stepper_and_data(crps_training=crps_training)
    if use_optimization:
        optimization_config = OptimizationConfig(
            optimizer_type="Adam",
            lr=0.0001,
        )
        optimization: OptimizationABC = optimization_config.build(
            stepper.modules, max_epochs=1
        )
    else:
        optimization = NullOptimization()
    result1 = stepper.train_on_batch(data, optimization)
    result2 = stepper.train_on_batch(data, optimization)
    output_dict = get_train_outputs_tensor_dict(result1, result2)
    filename = f"testdata/stepper_train_on_batch_regression-{use_optimization}.pt"
    if crps_training:
        filename = filename.replace(".pt", "-crps.pt")
    validate_tensor_dict(
        output_dict,
        os.path.join(
            DIR,
            filename,
        ),
    )


def test_stepper_predict_regression():
    torch.manual_seed(0)
    stepper, data = get_regression_stepper_and_data()
    initial_condition = data.get_start(
        prognostic_names=["b"],
        n_ic_timesteps=1,
    )
    output, next_state = stepper.predict(
        initial_condition, data, compute_derived_variables=True
    )
    output_dict = get_predict_output_tensor_dict(output, next_state)
    validate_tensor_dict(
        output_dict,
        os.path.join(DIR, f"testdata/stepper_predict_regression.pt"),
    )


def get_predict_output_tensor_dict(
    output: BatchData, next_state: PrognosticState
) -> dict[str, torch.Tensor]:
    return flatten_dict(
        {
            "output": output.data,
            "next_state": next_state.as_batch_data().data,
        }
    )


def get_train_outputs_tensor_dict(
    step_1: TrainOutput, step_2: TrainOutput
) -> dict[str, torch.Tensor]:
    return flatten_dict(
        {
            "step_1": _get_train_output_tensor_dict(step_1),
            "step_2": _get_train_output_tensor_dict(step_2),
        }
    )


def flatten_dict(
    d: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return_dict = {}
    for k, v in d.items():
        for k2, v2 in v.items():
            return_dict[f"{k}.{k2}"] = v2
    return return_dict


def _get_train_output_tensor_dict(data: TrainOutput) -> dict[str, torch.Tensor]:
    return_dict = {}
    for k, v in data.metrics.items():
        return_dict[f"metrics.{k}"] = v
    for k, v in data.gen_data.items():
        return_dict[f"gen_data.{k}"] = v
    for k, v in data.target_data.items():
        assert v.shape[1] == 1
        return_dict[f"target_data.{k}"] = v
    return return_dict


def test_set_train_eval():
    stepper = _get_stepper(["a"], ["a"])
    for module in stepper.modules:
        assert module.training
    stepper.set_eval()
    for module in stepper.modules:
        assert not module.training
    stepper.set_train()
    for module in stepper.modules:
        assert module.training


def test_get_serialized_stepper_vertical_coordinate():
    stepper = _get_stepper(["a"], ["a"])
    state = stepper.get_state()
    vertical_coordinate = get_serialized_stepper_vertical_coordinate(state)
    assert isinstance(vertical_coordinate, VerticalCoordinate)


def _get_stepper_with_input_masking(dataset_info_has_mask_provider: bool = True):
    # basic StepperConfig with input_masking configured
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt", config={"module": torch.nn.Identity()}
                    ),
                    in_names=["a"],
                    out_names=["a"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={"a": 0.0},
                            stds={"a": 1.0},
                        ),
                    ),
                )
            ),
        ),
        input_masking=StaticMaskingConfig(mask_value=0, fill_value=0.0),
    )
    mask_provider: MaskProvider | None = None
    if dataset_info_has_mask_provider:
        mask_provider = MaskProvider()
    return config.get_stepper(get_dataset_info(mask_provider=mask_provider))


def test_get_stepper_with_input_masking():
    # check that no error is raised when building a stepper with input_masking
    # configured when the vertical coordinate is a mask_provider

    # no error raised
    _ = _get_stepper_with_input_masking(dataset_info_has_mask_provider=True)


def test_get_stepper_with_input_masking_raises():
    # no get_mask_tensor_for method on vertical coordinate raises error when
    # input_masking provided in config
    with pytest.raises(MissingDatasetInfo, match="mask_provider"):
        _ = _get_stepper_with_input_masking(dataset_info_has_mask_provider=False)
