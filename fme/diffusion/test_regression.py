import dataclasses
import datetime
import os
from collections import namedtuple
from typing import Dict, Mapping, Tuple

import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.stepper.single_module import TrainOutput
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.device import get_device
from fme.core.generics.optimization import OptimizationABC
from fme.core.gridded_ops import LatLonOperations
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import NullOptimization, OptimizationConfig
from fme.core.rand import use_cpu_randn
from fme.core.testing.regression import validate_tensor_dict
from fme.diffusion.registry import ModuleSelector
from fme.diffusion.registry.sfno import ConditionalSFNOBuilder
from fme.diffusion.stepper import DiffusionStepper, DiffusionStepperConfig

SphericalData = namedtuple("SphericalData", ["data", "area_weights", "vertical_coord"])
TIMESTEP = datetime.timedelta(hours=6)
DEVICE = fme.get_device()

DIR = os.path.dirname(os.path.abspath(__file__))


def get_train_outputs_tensor_dict(
    step_1: TrainOutput, step_2: TrainOutput
) -> Dict[str, torch.Tensor]:
    return flatten_dict(
        {
            "step_1": _get_train_output_tensor_dict(step_1),
            "step_2": _get_train_output_tensor_dict(step_2),
        }
    )


def flatten_dict(
    d: Mapping[str, Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    return_dict = {}
    for k, v in d.items():
        for k2, v2 in v.items():
            return_dict[f"{k}.{k2}"] = v2
    return return_dict


def _get_train_output_tensor_dict(data: TrainOutput) -> Dict[str, torch.Tensor]:
    return_dict = {}
    for k, v in data.metrics.items():
        return_dict[f"metrics.{k}"] = v
    for k, v in data.gen_data.items():
        return_dict[f"gen_data.{k}"] = v
    for k, v in data.target_data.items():
        return_dict[f"target_data.{k}"] = v
    return return_dict


def get_predict_output_tensor_dict(
    output: BatchData, next_state: PrognosticState
) -> Dict[str, torch.Tensor]:
    return flatten_dict(
        {
            "output": output.data,
            "next_state": next_state.as_batch_data().data,
        }
    )


def get_regression_stepper_and_data(
    n_forward_steps: int = 1,
) -> Tuple[DiffusionStepper, BatchData]:
    in_names = ["a", "b"]
    out_names = ["b", "c"]
    n_samples = 3
    img_shape = (9, 18)
    device = get_device()
    all_names = list(set(in_names + out_names))
    area = torch.ones(img_shape)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    config = DiffusionStepperConfig(
        builder=ModuleSelector(
            type="ConditionalSFNO",
            config=dataclasses.asdict(
                ConditionalSFNOBuilder(
                    embed_dim=12,
                    num_layers=2,
                )
            ),
        ),
        in_names=in_names,
        out_names=out_names,
        normalization=NormalizationConfig(
            means={n: np.array([0.0], dtype=np.float32) for n in all_names},
            stds={n: np.array([1.1e-5], dtype=np.float32) for n in all_names},
        ),
        ocean=None,
    )
    stepper = config.get_stepper(
        img_shape, LatLonOperations(area), vertical_coordinate, TIMESTEP
    )
    data = BatchData(
        data={
            "a": torch.randn(n_samples, n_forward_steps + 1, *img_shape).to(device)
            * 1e-5,
            "b": torch.randn(n_samples, n_forward_steps + 1, *img_shape).to(device)
            * 1e-5,
            "c": torch.randn(n_samples, n_forward_steps + 1, *img_shape).to(device)
            * 1e-5,
        },
        time=xr.DataArray(
            np.zeros((n_samples, n_forward_steps + 1)),
            dims=["sample", "time"],
        ),
    )
    return stepper, data


def test_step_regression():
    torch.manual_seed(0)
    stepper, data = get_regression_stepper_and_data()
    with use_cpu_randn():
        output = stepper.step(
            {k: v[:, 0, ...] for k, v in data.data.items()},
            {},
        )
    assert len(output) > 0
    validate_tensor_dict(
        output,
        os.path.join(DIR, f"testdata/stepper_step_regression.pt"),
    )


@pytest.mark.parametrize(
    "use_optimization",
    [
        pytest.param(True, id="optimization"),
        pytest.param(False, id="no_optimization"),
    ],
)
def test_stepper_train_on_batch_regression(use_optimization: bool):
    torch.manual_seed(0)
    stepper, data = get_regression_stepper_and_data()
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
    with use_cpu_randn():
        result1 = stepper.train_on_batch(data, optimization)
        result2 = stepper.train_on_batch(data, optimization)
    output_dict = get_train_outputs_tensor_dict(result1, result2)
    filename = f"testdata/stepper_train_on_batch_regression-{use_optimization}.pt"
    validate_tensor_dict(
        output_dict,
        os.path.join(
            DIR,
            filename,
        ),
    )


def test_stepper_predict_regression():
    torch.manual_seed(0)
    stepper, data = get_regression_stepper_and_data(n_forward_steps=2)
    initial_condition = data.get_start(
        prognostic_names=["b"],
        n_ic_timesteps=1,
    )
    with use_cpu_randn():
        output, next_state = stepper.predict(
            initial_condition, data, compute_derived_variables=True
        )
    output_dict = get_predict_output_tensor_dict(output, next_state)
    validate_tensor_dict(
        output_dict,
        os.path.join(DIR, f"testdata/stepper_predict_regression.pt"),
    )
