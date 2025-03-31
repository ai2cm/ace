import datetime
import pathlib
from typing import List, Optional, Tuple

import pytest
import torch

import fme
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import LatLonOperations
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.typing_ import TensorDict

from .radiation import SeparateRadiationStepConfig


def get_tensor_dict(
    names: List[str], img_shape: Tuple[int, ...], n_samples: int
) -> TensorDict:
    data_dict = {}
    device = fme.get_device()
    for name in names:
        data_dict[name] = torch.rand(
            n_samples,
            *img_shape,
            device=device,
        )
    return data_dict


def get_network_and_loss_normalization_config(
    names: List[str],
    dir: Optional[pathlib.Path] = None,
) -> NetworkAndLossNormalizationConfig:
    if dir is None:
        return NetworkAndLossNormalizationConfig(
            network=NormalizationConfig(
                means={name: 0.0 for name in names},
                stds={name: 1.0 for name in names},
            ),
        )
    else:
        return NetworkAndLossNormalizationConfig(
            network=NormalizationConfig(
                global_means_path=dir / "means.nc",
                global_stds_path=dir / "stds.nc",
            ),
        )


def get_separate_radiation_config(
    dir: Optional[pathlib.Path] = None,
) -> SeparateRadiationStepConfig:
    normalization = get_network_and_loss_normalization_config(
        names=[
            "prog_a",
            "prog_b",
            "forcing_shared",
            "forcing_rad",
            "diagnostic_rad",
            "diagnostic_main",
        ],
        dir=dir,
    )

    return SeparateRadiationStepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={
                "scale_factor": 1,
                "embed_dim": 4,
                "num_layers": 2,
            },
        ),
        radiation_builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={
                "scale_factor": 1,
                "embed_dim": 4,
                "num_layers": 2,
            },
        ),
        main_prognostic_names=["prog_a", "prog_b"],
        shared_forcing_names=["forcing_shared"],
        radiation_only_forcing_names=["forcing_rad"],
        radiation_diagnostic_names=["diagnostic_rad"],
        main_diagnostic_names=["diagnostic_main"],
        normalization=normalization,
    )


@pytest.mark.parametrize("detach_radiation", [True, False])
def test_detach_radiation(detach_radiation: bool):
    config = get_separate_radiation_config()
    config.detach_radiation = detach_radiation
    device = fme.get_device()
    img_shape = (4, 2)
    area = torch.ones(img_shape, device=device)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    dataset_info = DatasetInfo(
        img_shape=img_shape,
        gridded_operations=LatLonOperations(area),
        vertical_coordinate=vertical_coordinate,
        timestep=datetime.timedelta(hours=6),
    )
    step = config.get_step(dataset_info)
    input_data = get_tensor_dict(
        names=["prog_a", "prog_b", "forcing_shared", "forcing_rad"],
        img_shape=img_shape,
        n_samples=1,
    )
    input_data["forcing_rad"].requires_grad = True
    output_data = step.step(input_data, input_data)
    for name, value in output_data.items():
        assert value.requires_grad, f"{name} should require grad"
    grad = torch.autograd.grad(
        outputs=output_data["diagnostic_rad"].sum(),
        inputs=input_data["forcing_rad"],
        allow_unused=True,
    )[0]
    assert grad is not None
    # have to call again as torch.autograd.grad frees the graph
    output_data = step.step(input_data, input_data)
    grad = torch.autograd.grad(
        outputs=output_data["diagnostic_main"].sum(),
        inputs=input_data["forcing_rad"],
        allow_unused=True,
    )[0]
    if detach_radiation:
        assert grad is None
    else:
        assert grad is not None
