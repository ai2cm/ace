import contextlib
from collections import namedtuple
from typing import Dict, Iterable, Optional, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import fme
from fme.core.aggregator.inference.main import InferenceAggregator
from fme.core.aggregator.null import NullAggregator
from fme.core.data_loading.typing import SigmaCoordinates
from fme.core.device import get_device
from fme.core.loss import ConservationLoss, get_dry_air_nonconservation
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.packer import Packer
from fme.core.prescriber import NullPrescriber, Prescriber, PrescriberConfig
from fme.core.stepper import (
    SingleModuleStepper,
    SingleModuleStepperConfig,
    _force_conserve_dry_air,
    run_on_batch,
)
from fme.fcn_training.registry import ModuleSelector

SphericalData = namedtuple("SphericalData", ["data", "area_weights", "sigma_coords"])


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
    return {n: torch.tensor([value], device=fme.get_device()) for n in names}


@pytest.mark.parametrize(
    "in_names,out_names,prescriber_config,expected_all_names",
    [
        (["a"], ["b"], None, ["a", "b"]),
        (["a"], ["a", "b"], None, ["a", "b"]),
        (["a", "b"], ["b"], None, ["a", "b"]),
        (["a", "b"], ["a", "b"], None, ["a", "b"]),
        (["a", "b"], ["a", "b"], PrescriberConfig("a", "mask", 0), ["a", "b", "mask"]),
        (["a", "b"], ["a", "b"], PrescriberConfig("a", "b", 0), ["a", "b"]),
    ],
)
def test_stepper_config_all_names_property(
    in_names, out_names, prescriber_config, expected_all_names
):
    config = SingleModuleStepperConfig(
        builder=MagicMock(),
        in_names=in_names,
        out_names=out_names,
        normalization=MagicMock(),
        prescriber=prescriber_config,
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
    conservation_loss = ConservationLoss(
        dry_air_penalty=None,
        area_weights=area,
        sigma_coordinates=sigma_coordinates,
    )
    stepped = run_on_batch(
        data=data,
        module=torch.nn.Identity(),
        normalizer=StandardNormalizer(
            means=get_scalar_data(["a", "b"], 0.0),
            stds=get_scalar_data(["a", "b"], 1.0),
        ),
        in_packer=Packer(["a", "b"]),
        out_packer=Packer(["a", "b"]),
        optimization=MagicMock(),
        loss_obj=torch.nn.MSELoss(),
        prescriber=NullPrescriber(),
        n_forward_steps=1,
        aggregator=MagicMock(),
        area=area,
        sigma_coordinates=sigma_coordinates,
        conservation_loss=conservation_loss,
    )
    assert torch.allclose(
        stepped.gen_data["a"], stepped.gen_data_norm["a"]
    )  # as std=1, mean=0, no change
    stepped_double_std = run_on_batch(
        data=data,
        module=torch.nn.Identity(),
        normalizer=StandardNormalizer(
            means=get_scalar_data(["a", "b"], 0.0),
            stds=get_scalar_data(["a", "b"], 2.0),
        ),
        in_packer=Packer(["a", "b"]),
        out_packer=Packer(["a", "b"]),
        optimization=MagicMock(),
        loss_obj=torch.nn.MSELoss(),
        prescriber=NullPrescriber(),
        n_forward_steps=1,
        aggregator=MagicMock(),
        area=area,
        sigma_coordinates=sigma_coordinates,
        conservation_loss=conservation_loss,
    )
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
        stepped.metrics["loss"], 4.0 * stepped_double_std.metrics["loss"], rtol=1e-4
    )  # mse scales with std**2


def test_run_on_batch_addition_series():
    torch.manual_seed(0)

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    n_steps = 4
    data = get_data(["a", "b"], n_samples=5, n_time=n_steps + 1).data
    area = torch.ones((5, 5), device=fme.get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    conservation_loss = ConservationLoss(
        dry_air_penalty=None,
        area_weights=area,
        sigma_coordinates=sigma_coordinates,
    )
    stepped = run_on_batch(
        data=data,
        module=AddOne(),
        normalizer=StandardNormalizer(
            means=get_scalar_data(["a", "b"], 0.0),
            stds=get_scalar_data(["a", "b"], 1.0),
        ),
        in_packer=Packer(["a", "b"]),
        out_packer=Packer(["a", "b"]),
        optimization=MagicMock(),
        loss_obj=torch.nn.MSELoss(),
        prescriber=NullPrescriber(),
        n_forward_steps=n_steps,
        aggregator=MagicMock(),
        area=area,
        sigma_coordinates=sigma_coordinates,
        conservation_loss=conservation_loss,
    )
    assert stepped.gen_data["a"].shape == (5, n_steps + 1, 5, 5)
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
    assert torch.allclose(stepped.gen_data["a"][:, 0], data["a"][:, 0])
    assert torch.allclose(stepped.gen_data["b"][:, 0], data["b"][:, 0])
    assert torch.allclose(
        stepped.gen_data_norm["a"][:, 0], stepped.target_data_norm["a"][:, 0]
    )
    assert torch.allclose(
        stepped.gen_data_norm["b"][:, 0], stepped.target_data_norm["b"][:, 0]
    )


def test_run_on_batch_with_prescriber():
    torch.manual_seed(0)

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    n_steps = 3
    data = get_data(["a", "b", "mask"], n_samples=5, n_time=n_steps + 1).data
    data["mask"] = torch.zeros_like(data["mask"], dtype=torch.int)
    data["mask"][:, :, :, 0] = 1
    stds = {
        "a": torch.tensor([2.0], device=fme.get_device()),
        "b": torch.tensor([3.0], device=fme.get_device()),
        "mask": torch.tensor([1.0], device=fme.get_device()),
    }
    area = torch.ones((5, 5), device=fme.get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    conservation_loss = ConservationLoss(
        dry_air_penalty=None,
        area_weights=area,
        sigma_coordinates=sigma_coordinates,
    )
    stepped = run_on_batch(
        data=data,
        module=AddOne(),
        normalizer=StandardNormalizer(
            means=get_scalar_data(["a", "b", "mask"], 0.0),
            stds=stds,
        ),
        in_packer=Packer(["a", "b"]),
        out_packer=Packer(["a", "b"]),
        optimization=MagicMock(),
        loss_obj=torch.nn.MSELoss(),
        n_forward_steps=n_steps,
        prescriber=Prescriber("b", "mask", 1),
        aggregator=MagicMock(),
        area=area,
        sigma_coordinates=sigma_coordinates,
        conservation_loss=conservation_loss,
    )
    for i in range(n_steps):
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
        shapes=shapes,
        area=area,
        sigma_coordinates=sigma_coordinates,
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
    data: Dict[str, torch.Tensor],
    in_names,
    out_names,
    prescriber_config: Optional[PrescriberConfig],
    n_forward_steps,
    optimization_config: Optional[OptimizationConfig],
    aggregator: Union[NullAggregator, InferenceAggregator],
):
    """Sets up the requisite classes to run run_on_batch."""
    module = ReturnZerosModule(len(in_names), len(out_names))

    if optimization_config is None:
        optimization: Union[NullOptimization, Optimization] = NullOptimization()
    else:
        optimization = optimization_config.build(module.parameters(), 2)

    if aggregator is None:
        aggregator = NullAggregator()

    if prescriber_config is None:
        prescriber = NullPrescriber()
    else:
        prescriber = prescriber_config.build(in_names, out_names)

    area = torch.ones((5, 5), device=fme.get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    conservation_loss = ConservationLoss(
        dry_air_penalty=None,
        area_weights=area,
        sigma_coordinates=sigma_coordinates,
    )
    stepped = run_on_batch(
        data=data,
        module=module,
        normalizer=StandardNormalizer(
            means=get_scalar_data(in_names, 0.0),
            stds=get_scalar_data(in_names, 1.0),
        ),
        in_packer=Packer(in_names),
        out_packer=Packer(out_names),
        optimization=optimization,
        loss_obj=torch.nn.MSELoss(),
        n_forward_steps=n_forward_steps,
        prescriber=prescriber,
        aggregator=aggregator,
        area=area,
        sigma_coordinates=sigma_coordinates,
        conservation_loss=conservation_loss,
    )

    return stepped


@pytest.mark.parametrize(
    "is_input,is_output,is_prescribed,time_dim_needed_for_var",
    [
        pytest.param(True, True, True, True, id="in_out_prescribed"),
        pytest.param(True, True, False, False, id="in_out_not_prescribed"),
        pytest.param(False, True, False, False, id="out_only_not_prescribed"),
    ],
)
@pytest.mark.parametrize("n_forward_steps", [1, 2, 3], ids=lambda p: f"k={p}")
@pytest.mark.parametrize("use_aggregator", [True, False], ids=["use_agg", ""])
@pytest.mark.parametrize("is_train", [True, False], ids=["is_train", ""])
@pytest.mark.parametrize("is_ragged_time_dim", [True, False], ids=["is_ragged", ""])
def test_run_on_batch(
    n_forward_steps,
    is_input,
    is_output,
    is_train,
    is_prescribed,
    time_dim_needed_for_var,
    use_aggregator,
    is_ragged_time_dim,
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
        prescriber = PrescriberConfig("b", mask_name, 0)
    else:
        prescriber = None

    data, area_weights, sigma_coords = get_data(all_names, 3, n_forward_steps + 1)
    time_dim = 1

    if is_ragged_time_dim:
        # make time_dim ragged: only keep t=0, while preserving `time_dim`
        data["b"] = data["b"].select(dim=time_dim, index=0).unsqueeze(time_dim)

    if use_aggregator:
        aggregator: Union[InferenceAggregator, NullAggregator] = InferenceAggregator(
            area_weights, sigma_coords, n_timesteps=n_forward_steps + 1
        )
    else:
        aggregator = NullAggregator()

    if is_train:
        optimization = OptimizationConfig()
    else:
        optimization = None

    if is_ragged_time_dim and (is_train or time_dim_needed_for_var):
        context = pytest.raises(ValueError)
    else:
        context = contextlib.nullcontext()

    with context:
        _setup_and_run_on_batch(
            data,
            in_names,
            out_names,
            prescriber,
            n_forward_steps,
            optimization,
            aggregator,
        )


def test_force_conserve_dry_air():
    torch.random.manual_seed(0)
    data = {
        "PRESsfc": 10.0 + torch.rand(size=(3, 2, 5, 5)),
        "specific_total_water_0": torch.rand(size=(3, 2, 5, 5)),
        "specific_total_water_1": torch.rand(size=(3, 2, 5, 5)),
    }
    sigma_coordinates = SigmaCoordinates(
        ak=torch.asarray([3.0, 1.0, 0.0]), bk=torch.asarray([0.0, 0.6, 1.0])
    )
    area_weights = 1.0 + torch.rand(size=(5, 5))
    original_nonconservation = get_dry_air_nonconservation(
        data,
        sigma_coordinates=sigma_coordinates,
        area_weights=area_weights,
    )
    assert original_nonconservation > 0.0
    in_data = {k: v.select(dim=1, index=0) for k, v in data.items()}
    out_data = {k: v.select(dim=1, index=1) for k, v in data.items()}
    fixed_out_data = _force_conserve_dry_air(
        in_data,
        out_data,
        sigma_coordinates=sigma_coordinates,
        area=area_weights,
    )
    new_data = {
        k: torch.stack([v, fixed_out_data[k]], dim=1) for k, v in in_data.items()
    }
    new_nonconservation = get_dry_air_nonconservation(
        new_data,
        sigma_coordinates=sigma_coordinates,
        area_weights=area_weights,
    )
    assert new_nonconservation < original_nonconservation
    np.testing.assert_almost_equal(new_nonconservation.cpu().numpy(), 0.0, decimal=6)
