import dataclasses
import datetime
import pathlib

import torch

from fme.ace.stepper.single_module import StepperConfig
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.step import SingleModuleStepConfig, StepSelector

TIMESTEP = datetime.timedelta(hours=6)


class _PlusOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def save_stepper_checkpoint(
    path: pathlib.Path,
    in_names: list[str] | None = None,
    out_names: list[str] | None = None,
) -> StepperConfig:
    """Save a minimal stepper checkpoint and return the config used.

    Args:
        path: File path at which to save the checkpoint.
        in_names: Input variable names. Defaults to ["a", "b"].
        out_names: Output variable names. Defaults to ["a"].

    Returns:
        The StepperConfig used to create the saved stepper.
    """
    if in_names is None:
        in_names = ["a", "b"]
    if out_names is None:
        out_names = ["a"]
    all_names = list(set(in_names) | set(out_names))
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt", config={"module": _PlusOne()}
                    ),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={name: 0.0 for name in all_names},
                            stds={name: 1.0 for name in all_names},
                        ),
                    ),
                ),
            ),
        ),
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(4), lon=torch.zeros(8)
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(4), bk=torch.arange(4)
        ),
        timestep=TIMESTEP,
        variable_metadata={
            out_names[0]: VariableMetadata(units="K", long_name="temperature"),
        },
    )
    stepper = config.get_stepper(dataset_info=dataset_info)
    torch.save({"stepper": stepper.get_state()}, path)
    return config
