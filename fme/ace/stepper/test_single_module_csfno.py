"""
Parallel regression tests for the SingleModuleStepper with NoiseConditionedSFNO.

These tests verify that the forward pass and loss computation produce identical
results regardless of spatial decomposition (nproc=1 vs model-parallel).
"""

import dataclasses
import datetime
import os
from collections.abc import Mapping

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.ace.stepper.single_module import (
    StepperConfig,
    TrainOutput,
    TrainStepper,
    TrainStepperConfig,
)
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed.distributed import Distributed
from fme.core.loss import StepLossConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.optimization import NullOptimization, OptimizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.step import SingleModuleStepConfig, StepSelector
from fme.core.testing.regression import validate_tensor_dict
from fme.core.typing_ import EnsembleTensorDict

DIR = os.path.abspath(os.path.dirname(__file__))
TIMESTEP = datetime.timedelta(hours=6)


def get_dataset_info(
    img_shape=(5, 5),
) -> DatasetInfo:
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(img_shape[-2]),
        lon=torch.zeros(img_shape[-1]),
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    return DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=TIMESTEP,
    )


def _get_train_stepper(
    stepper_config: StepperConfig,
    dataset_info: DatasetInfo,
    **train_config_kwargs,
) -> TrainStepper:
    train_config = TrainStepperConfig(**train_config_kwargs)
    return train_config.get_train_stepper(stepper_config, dataset_info)


def get_regression_stepper_and_data() -> (
    tuple[TrainStepper, BatchData, tuple[int, int]]
):
    in_names = ["a", "b"]
    out_names = ["b", "c"]
    n_forward_steps = 2
    n_samples = 3
    img_shape = (9, 18)
    device = get_device()

    all_names = list(set(in_names + out_names))

    loss = StepLossConfig(type="AreaWeightedMSE")

    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="NoiseConditionedSFNO",
                        config=dataclasses.asdict(
                            NoiseConditionedSFNOBuilder(
                                embed_dim=16,
                                num_layers=2,
                                noise_embed_dim=16,
                                noise_type="isotropic",
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
    )

    dataset_info = get_dataset_info(img_shape=img_shape)
    train_stepper = _get_train_stepper(config, dataset_info, loss=loss)
    data = BatchData.new_on_device(
        data={
            "a": torch.randn(n_samples, n_forward_steps + 1, *img_shape).to(device),
            "b": torch.randn(n_samples, n_forward_steps + 1, *img_shape).to(device),
            "c": torch.randn(n_samples, n_forward_steps + 1, *img_shape).to(device),
        },
        time=xr.DataArray(
            np.zeros((n_samples, n_forward_steps + 1)),
            dims=["sample", "time"],
        ),
        labels=None,
        epoch=0,
        horizontal_dims=["lat", "lon"],
    )
    data = data.scatter_spatial(img_shape)
    return train_stepper, data, img_shape


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


def get_train_outputs_tensor_dict(
    step_1: TrainOutput, step_2: TrainOutput
) -> dict[str, torch.Tensor]:
    return flatten_dict(
        {
            "step_1": _get_train_output_tensor_dict(step_1),
            "step_2": _get_train_output_tensor_dict(step_2),
        }
    )


@pytest.mark.parallel
def test_stepper_train_on_batch_regression():
    torch.manual_seed(0)
    train_stepper, data, img_shape = get_regression_stepper_and_data()
    optimization = NullOptimization()
    result1 = train_stepper.train_on_batch(data, optimization)
    result2 = train_stepper.train_on_batch(data, optimization)
    dist = Distributed.get_instance()
    for result in [result1, result2]:
        result.gen_data = EnsembleTensorDict(
            dist.gather_spatial(dict(result.gen_data), img_shape)
        )
        result.target_data = EnsembleTensorDict(
            dist.gather_spatial(dict(result.target_data), img_shape)
        )
    output_dict = get_train_outputs_tensor_dict(result1, result2)
    validate_tensor_dict(
        output_dict,
        os.path.join(
            DIR,
            "testdata/csfno_stepper_train_on_batch_regression.pt",
        ),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parallel
def test_stepper_train_on_batch_with_optimization_regression():
    torch.manual_seed(0)
    train_stepper, data, img_shape = get_regression_stepper_and_data()
    optimization = OptimizationConfig(
        optimizer_type="Adam",
        lr=0.0001,
    ).build(train_stepper.modules, max_epochs=1)
    result1 = train_stepper.train_on_batch(data, optimization)
    result2 = train_stepper.train_on_batch(data, optimization)
    dist = Distributed.get_instance()
    for result in [result1, result2]:
        result.gen_data = EnsembleTensorDict(
            dist.gather_spatial(dict(result.gen_data), img_shape)
        )
        result.target_data = EnsembleTensorDict(
            dist.gather_spatial(dict(result.target_data), img_shape)
        )
    output_dict = get_train_outputs_tensor_dict(result1, result2)
    validate_tensor_dict(
        output_dict,
        os.path.join(
            DIR,
            "testdata/csfno_stepper_train_on_batch_with_optimization_regression.pt",
        ),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parallel
def test_stepper_predict_regression():
    torch.manual_seed(0)
    train_stepper, data, img_shape = get_regression_stepper_and_data()
    stepper = train_stepper._stepper
    initial_condition = data.get_start(
        prognostic_names=["b"],
        n_ic_timesteps=1,
    )
    output, next_state = stepper.predict(
        initial_condition, data, compute_derived_variables=True
    )
    dist = Distributed.get_instance()
    output_data = dist.gather_spatial(dict(output.data), img_shape)
    next_state_data = dist.gather_spatial(
        dict(next_state.as_batch_data().data), img_shape
    )
    output_dict = flatten_dict({"output": output_data, "next_state": next_state_data})
    validate_tensor_dict(
        output_dict,
        os.path.join(DIR, "testdata/csfno_stepper_predict_regression.pt"),
        atol=1e-4,
        rtol=1e-4,
    )
