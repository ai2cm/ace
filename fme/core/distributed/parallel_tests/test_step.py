"""
This file contains regression tests for StepABC implementations,
ensuring they produce the same results regardless of parallel decomposition.

It also includes some basic tests, but only on implementations we expect to
work in parallel mode. This may duplicate StepABC test coverage in the
non-parallel StepABC tests (which include also cases that don't work in parallel).
"""

import dataclasses
import datetime
import pathlib
import tempfile
import unittest
import unittest.mock
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

import fme
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.ace.testing.fv3gfs_data import get_scalar_dataset
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig, EnergyBudgetConfig
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed.distributed import Distributed
from fme.core.distributed.non_distributed import DummyWrapper
from fme.core.labels import BatchLabels
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.optimization import Optimization, OptimizationConfig, SchedulerConfig
from fme.core.registry import ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.multi_call import MultiCallConfig, MultiCallStepConfig
from fme.core.step.secondary_decoder import SecondaryDecoderConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepABC, StepSelector
from fme.core.typing_ import TensorDict

DEFAULT_IMG_SHAPE = (45, 90)

DATA_DIR = pathlib.Path(__file__).parent / "testdata"


def get_network_and_loss_normalization_config(
    names: list[str],
    dir: pathlib.Path | None = None,
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


def get_single_module_noise_conditioned_selector(
    dir: pathlib.Path | None = None,
) -> StepSelector:
    normalization = get_network_and_loss_normalization_config(
        names=[
            "forcing_shared",
            "forcing_rad",
            "diagnostic_main",
            "diagnostic_rad",
        ],
        dir=dir,
    )
    return StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="NoiseConditionedSFNO",
                    config=dataclasses.asdict(
                        NoiseConditionedSFNOBuilder(
                            embed_dim=4,
                            noise_embed_dim=4,
                            noise_type="isotropic",
                            filter_type="linear",
                            filter_num_groups=2,
                            context_pos_embed_dim=2,
                            pos_embed=False,
                            num_layers=2,
                            local_blocks=[0],
                            affine_norms=True,
                        )
                    ),
                ),
                in_names=["forcing_shared", "forcing_rad"],
                out_names=["diagnostic_main"],
                secondary_decoder=SecondaryDecoderConfig(
                    secondary_diagnostic_names=["diagnostic_rad"],
                    network=ModuleSelector(type="MLP", config={}),
                ),
                normalization=normalization,
            ),
        ),
    )


def get_single_module_with_atmosphere_corrector_selector(
    dir: pathlib.Path | None = None,
) -> StepSelector:
    in_names = [
        "DSWRFtoa",
        "HGTsfc",
        "air_temperature_0",
        "air_temperature_1",
        "air_temperature_2",
        "air_temperature_3",
        "air_temperature_4",
        "air_temperature_5",
        "specific_total_water_0",
        "specific_total_water_1",
        "specific_total_water_2",
        "specific_total_water_3",
        "specific_total_water_4",
        "specific_total_water_5",
        "PRESsfc",
        "PRATEsfc",
    ]
    out_names = [
        "air_temperature_0",
        "air_temperature_1",
        "air_temperature_2",
        "air_temperature_3",
        "air_temperature_4",
        "air_temperature_5",
        "specific_total_water_0",
        "specific_total_water_1",
        "specific_total_water_2",
        "specific_total_water_3",
        "specific_total_water_4",
        "specific_total_water_5",
        "PRESsfc",
        "PRATEsfc",
        "LHTFLsfc",
        "SHTFLsfc",
        "ULWRFsfc",
        "ULWRFtoa",
        "DLWRFsfc",
        "DSWRFsfc",
        "USWRFtoa",
        "USWRFsfc",
        "tendency_of_total_water_path_due_to_advection",
    ]
    normalization = get_network_and_loss_normalization_config(
        names=list(set(in_names).union(out_names)),
        dir=dir,
    )
    return StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="NoiseConditionedSFNO",
                    config=dataclasses.asdict(
                        NoiseConditionedSFNOBuilder(
                            embed_dim=4,
                            noise_embed_dim=4,
                            noise_type="isotropic",
                            num_layers=2,
                            local_blocks=[0],
                        )
                    ),
                ),
                in_names=in_names,
                out_names=out_names,
                normalization=normalization,
                corrector=AtmosphereCorrectorConfig(
                    conserve_dry_air=True,
                    zero_global_mean_moisture_advection=True,
                    moisture_budget_correction="advection_and_precipitation",
                    force_positive_names=["PRATEsfc"],
                    total_energy_budget_correction=EnergyBudgetConfig(
                        "constant_temperature"
                    ),
                ),
            ),
        ),
    )


def get_multi_call_selector(
    dir: pathlib.Path | None = None,
) -> StepSelector:
    return StepSelector(
        type="multi_call",
        config=dataclasses.asdict(
            MultiCallStepConfig(
                wrapped_step=get_single_module_noise_conditioned_selector(dir),
                config=MultiCallConfig(
                    forcing_name="forcing_rad",
                    forcing_multipliers={"double": 2.0},
                    output_names=["diagnostic_rad"],
                ),
            ),
        ),
    )


SELECTOR_GETTERS = {
    "sm_with_atmos_corr": get_single_module_with_atmosphere_corrector_selector,
    "sm_noise_conditioned": get_single_module_noise_conditioned_selector,
    "multi_call": get_multi_call_selector,
}

SELECTOR_CONFIG_CASES = [
    pytest.param(
        getter(),
        id=getter.__name__,
    )
    for getter in SELECTOR_GETTERS.values()
]
TIMESTEP = datetime.timedelta(hours=6)


def get_tensor_dict(
    names: list[str], img_shape: tuple[int, int], n_samples: int
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


def get_step(
    selector: StepSelector,
    img_shape: tuple[int, int],
    init_weights: Callable[[list[nn.Module]], None] = lambda _: None,
    all_labels: set[str] | None = None,
) -> StepABC:
    device = fme.get_device()
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(img_shape[0], device=device),
        lon=torch.zeros(img_shape[1], device=device),
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=TIMESTEP,
        all_labels=all_labels,
    )
    return selector.get_step(dataset_info, init_weights)


@pytest.mark.parametrize("config", SELECTOR_CONFIG_CASES)
@pytest.mark.parallel
def test_step_applies_wrapper(config: StepSelector):
    torch.manual_seed(0)
    img_shape = DEFAULT_IMG_SHAPE
    n_samples = 5
    step = get_step(config, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    dist = Distributed.get_instance()
    input_data = dist.scatter_spatial(input_data, img_shape)
    next_step_input_data = dist.scatter_spatial(next_step_input_data, img_shape)
    multi_calls = 1
    if isinstance(config._step_config_instance, MultiCallStepConfig):
        if config._step_config_instance.config is not None:
            multi_calls += len(config._step_config_instance.config.forcing_multipliers)

    wrapper = unittest.mock.MagicMock(side_effect=lambda x: x)
    step.step(
        args=StepArgs(
            input=input_data, next_step_input_data=next_step_input_data, labels=None
        ),
        wrapper=wrapper,
    )
    assert wrapper.call_count == multi_calls * len(step.modules)
    for module in step.modules:
        wrapper.assert_any_call(module)


@pytest.mark.parametrize("config", SELECTOR_CONFIG_CASES)
@pytest.mark.parallel
def test_step_initializes_weights(config: StepSelector):
    torch.manual_seed(0)
    img_shape = DEFAULT_IMG_SHAPE
    init_weights = unittest.mock.MagicMock(side_effect=lambda x: x)
    step = get_step(config, img_shape, init_weights)
    assert init_weights.called
    call_args, call_kwargs = init_weights.call_args
    assert len(call_args) == 1
    assert len(call_kwargs) == 0
    assert isinstance(call_args[0], list | nn.ModuleList)
    assert len(call_args[0]) == len(step.modules)
    for i, module in enumerate(step.modules):
        assert isinstance(
            module, DummyWrapper | torch.nn.parallel.DistributedDataParallel
        )
        assert call_args[0][i] is module.module


@pytest.mark.parametrize(
    "get_config",
    SELECTOR_GETTERS.values(),
)
@pytest.mark.parallel
def test_load_config(
    get_config: Callable[[pathlib.Path | None], StepSelector],
):
    non_path_config: StepSelector = get_config(
        None
    )  # doesn't depend on files, use to get names
    all_names = set(non_path_config.input_names).union(non_path_config.output_names)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        get_scalar_dataset(all_names, fill_value=0.1).to_netcdf(temp_path / "means.nc")
        get_scalar_dataset(all_names, fill_value=1.1).to_netcdf(temp_path / "stds.nc")
        config = get_config(temp_path)
        config.load()
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(config, img_shape)
    normalizer = step.normalizer
    assert normalizer.means.keys() == all_names
    assert normalizer.stds.keys() == all_names
    assert all(normalizer.means[name] == 0.1 for name in all_names)
    assert all(normalizer.stds[name] == 1.1 for name in all_names)


@pytest.mark.parametrize(
    "get_config",
    SELECTOR_GETTERS.values(),
)
@pytest.mark.parallel
def test_load_is_required_for_path_config(
    get_config: Callable[[pathlib.Path | None], StepSelector],
):
    non_path_config: StepSelector = get_config(
        None
    )  # doesn't depend on files, use to get names
    all_names = set(non_path_config.input_names).union(non_path_config.output_names)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        get_scalar_dataset(all_names, fill_value=0.1).to_netcdf(temp_path / "means.nc")
        get_scalar_dataset(all_names, fill_value=1.1).to_netcdf(temp_path / "stds.nc")
        config = get_config(temp_path)
    img_shape = DEFAULT_IMG_SHAPE
    with pytest.raises(FileNotFoundError):
        get_step(config, img_shape)


def cache_step_input(
    step: StepABC,
    input_data: TensorDict,
    next_step_input_data: TensorDict,
    labels: BatchLabels | None,
    checkpoint_path: pathlib.Path,
):
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=fme.get_device())
        step.load_state(checkpoint["step_state_dict"])
        input_data = checkpoint["input_data"]
        next_step_input_data = checkpoint["next_step_input_data"]
        label_tensor = checkpoint["label_tensor"]
        if label_tensor is not None:
            assert isinstance(labels, BatchLabels)
            labels.tensor[:] = label_tensor
        return step, input_data, next_step_input_data, labels
    else:
        checkpoint = {
            "step_state_dict": step.get_state(),
            "input_data": input_data,
            "next_step_input_data": next_step_input_data,
            "label_tensor": labels.tensor if labels is not None else None,
        }
        torch.save(checkpoint, checkpoint_path)
        raise AssertionError(
            f"Step state checkpoint created at {checkpoint_path}, "
            "please re-run the test."
        )


def cache_step_output(output_data: TensorDict, checkpoint_path: pathlib.Path):
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=fme.get_device())
        expected_output = checkpoint["output_data"]
        for name in output_data.keys():
            torch.testing.assert_close(output_data[name], expected_output[name])
    else:
        checkpoint = {
            "output_data": output_data,
        }
        torch.save(checkpoint, checkpoint_path)
        raise AssertionError(
            f"Step output checkpoint created at {checkpoint_path}, "
            "please re-run the test."
        )


@pytest.mark.parametrize(
    "case_name,get_config",
    SELECTOR_GETTERS.items(),
)
@pytest.mark.parallel
def test_step_regression(
    case_name,
    get_config: Callable[[pathlib.Path | None], StepSelector],
):
    """
    Test that the step produces the same output as a regression target file.

    This ensures the step produce the same result regardless of parallel
    decomposition, as well as catching any unintended changes to the
    step's behavior.
    """
    dist = Distributed.get_instance()
    torch.manual_seed(0)
    img_shape = (20, 40)
    n_samples = 2
    selector = get_config(None)
    if selector.config.get("conditional", False):
        labels = BatchLabels.new_from_set(
            {"a", "b"}, n_samples=n_samples, device=fme.get_device()
        )
        labels.tensor[:] = torch.as_tensor(
            np.random.randint(0, 2, (n_samples,)), device=fme.get_device()
        )
    else:
        labels = None
    step = get_step(selector, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    step, input_data, next_step_input_data, labels = cache_step_input(
        step,
        input_data,
        next_step_input_data,
        labels,
        DATA_DIR / f"{case_name}_input.pt",
    )
    # Scatter global inputs to local spatial chunks
    input_data = dist.scatter_spatial(input_data, img_shape)
    next_step_input_data = dist.scatter_spatial(next_step_input_data, img_shape)

    output = step.step(
        args=StepArgs(
            input=input_data, next_step_input_data=next_step_input_data, labels=labels
        ),
        wrapper=lambda x: x,
    )

    # Gather local outputs back to global for comparison
    output = dist.gather_spatial(output, img_shape)

    cache_step_output(output, DATA_DIR / f"{case_name}_output.pt")

def _run_step_optimization_backward(
    img_shape: tuple[int, int],
    n_samples: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Run a single forward + backward through a Step using Optimization,
    and return:
      - scalar loss on this rank
      - gradients for all parameters (CPU tensors)
    """
    device = fme.get_device()
    dist = Distributed.get_instance()

    selector = get_single_module_noise_conditioned_selector(None)
    step = get_step(selector, img_shape)

    modules = nn.ModuleList(step.modules)
    opt_config = OptimizationConfig(
        optimizer_type="Adam",
        lr=1e-3,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
    )
    optimization = opt_config.build(modules, max_epochs=1)
    optimization.set_mode(modules)

    # Global inputs; Distributed backend decides what scatter_spatial does
    input_data: TensorDict = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_input_data: TensorDict = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )

    # Use the same scatter pattern as the step tests; in serial this is a no-op
    input_data = dist.scatter_spatial(input_data, img_shape)
    next_input_data = dist.scatter_spatial(next_input_data, img_shape)

    # Forward
    out = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_input_data,
            labels=None,
        ),
        wrapper=lambda x: x,
    )

    # Use the real training loss from the step output
    if isinstance(out, dict) and "loss" in out:
        loss = out["loss"]
    else:
        raise RuntimeError(
            "Step output does not contain 'loss'; "
            "wire this test to the real training loss output."
        )

    # Route loss through Optimization, but only run backward (no step/zero yet)
    optimization.accumulate_loss(loss)
    total_loss = optimization.get_accumulated_loss()
    optimization._backward(total_loss)

    # Collect parameter grads
    grads: dict[str, torch.Tensor] = {}
    for name, p in step.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().cpu().clone()

    return total_loss.detach().cpu(), grads


@pytest.mark.parallel
def test_step_optimization_backward_matches_baseline():
    """
    Regression test for Step+Optimization backward under spatial parallelism.

    Uses the same forward/Optimization path in both serial and spatial
    backends; serial run is used to create the baseline. Spatial backends
    must reproduce the baseline loss and parameter gradients element-wise.
    """
    DATA_DIR = pathlib.Path(__file__).parent / "testdata"
    BASELINE_FILE = DATA_DIR / "backward_with_opt_baseline.pt"

    dist = Distributed.get_instance()
    torch.manual_seed(0)

    img_shape = (20, 40)
    n_samples = 2

    loss, grads = _run_step_optimization_backward(img_shape, n_samples)

    # Only root rank writes/compares baseline
    if not dist.is_root():
        return

    if not BASELINE_FILE.exists():
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "img_shape": img_shape,
                "n_samples": n_samples,
                "loss": loss,
                "grads": grads,
            },
            BASELINE_FILE,
        )
        raise AssertionError(
            f"Baseline created at {BASELINE_FILE}. "
            "Re-run the test to perform regression check."
        )

    baseline = torch.load(BASELINE_FILE, map_location="cpu")
    assert tuple(baseline["img_shape"]) == tuple(img_shape)
    assert baseline["n_samples"] == n_samples

    baseline_loss = baseline["loss"]
    baseline_grads: dict[str, torch.Tensor] = baseline["grads"]

    # 1) Loss finite and close to baseline.
    assert torch.isfinite(loss), "Loss is not finite on this rank"

    actual_loss = loss.item()
    expected_loss = baseline_loss.item()
    rel_loss = abs(actual_loss - expected_loss) / max(abs(expected_loss), 1e-12)
    assert rel_loss < 1e-6, (
        f"Loss deviates from baseline: "
        f"actual={actual_loss:.8e}, expected={expected_loss:.8e}, "
        f"rel_diff={rel_loss:.3e}"
    )

    # 2) Parameter gradients match baseline element-wise.
    # Require same parameter set as baseline
    assert set(grads.keys()) == set(
        baseline_grads.keys()
    ), "Parameter set changed since baseline generation"

    for name in sorted(grads.keys()):
        g = grads[name]
        g_ref = baseline_grads[name]
        assert g.shape == g_ref.shape, f"Shape mismatch for grad '{name}'"
        diff = (g - g_ref).abs()
        max_abs = diff.max().item()
        max_rel = (diff / g_ref.abs().clamp_min(1e-12)).max().item()
        assert torch.allclose(g, g_ref, rtol=1e-6, atol=1e-7), (
            f"Gradient for '{name}' deviates from baseline: "
            f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
        )
