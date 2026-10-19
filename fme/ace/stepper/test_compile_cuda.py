"""End-to-end coverage of the default ``torch.compile`` path on CUDA.

The suite runs eagerly (see the root ``conftest.py``), and the other compile
tests substitute the ``aot_eager`` backend to stay within timeouts, so this
module is the one place a stepper trains and predicts through the inductor
backend that steps compile with by default on CUDA.
"""

import pytest
import torch

from fme.ace.stepper.single_module import TrainStepper
from fme.ace.stepper.test_single_module import get_regression_stepper_and_data
from fme.core.optimization import NullOptimization, OptimizationConfig
from fme.core.step.single_module import SingleModuleStep
from fme.core.testing import (
    compile_on,  # noqa: F401  autouse in this module
    dynamo_hygiene,  # noqa: F401  autouse in this module
)
from fme.core.typing_ import TensorMapping


def _assert_all_finite(tensors: TensorMapping) -> None:
    assert len(tensors) > 0
    for name, value in tensors.items():
        assert torch.isfinite(value).all(), name


def _get_compiled_step(train_stepper: TrainStepper) -> SingleModuleStep:
    step = train_stepper._stepper._step_obj
    assert isinstance(step, SingleModuleStep)
    assert step.module.is_compiled
    # _orig_mod is how torch._dynamo's OptimizedModule exposes the module
    # torch.compile was called on
    assert step.module.forward_module._orig_mod is step.module.torch_module
    return step


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="inductor default path runs on CUDA"
)
def test_stepper_trains_and_predicts_with_default_cuda_compile():
    """A stepper built through the training config compiles its step with the
    default (inductor) backend, and then trains, validates and predicts with
    finite outputs.
    """
    torch.manual_seed(0)
    train_stepper, data = get_regression_stepper_and_data()
    step = _get_compiled_step(train_stepper)
    optimization = OptimizationConfig(optimizer_type="Adam", lr=1e-4).build(
        train_stepper.modules, max_epochs=1
    )
    params_before = [p.detach().clone() for p in train_stepper.modules.parameters()]

    train_output = train_stepper.train_on_batch(data, optimization)

    _assert_all_finite(train_output.metrics)
    _assert_all_finite(train_output.gen_data)
    params_after = list(train_stepper.modules.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before, params_after)
    ), "the optimizer step did not update any parameter"
    for param in params_after:
        assert torch.isfinite(param).all()

    train_stepper.set_eval()
    with torch.no_grad():
        validation_output = train_stepper.train_on_batch(data, NullOptimization())
        initial_condition = data.get_start(prognostic_names=["b"], n_ic_timesteps=1)
        prediction, _ = train_stepper._stepper.predict(initial_condition, data)
    _assert_all_finite(validation_output.metrics)
    _assert_all_finite(validation_output.gen_data)
    _assert_all_finite(prediction.data)
    # the compiled forward was used rather than an eager fallback
    assert step.module.is_compiled
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
