"""Guards the default ``torch.compile`` path on CUDA.

The rest of the suite compiles whenever production does, so on CUDA it covers
compiled numerics; this module asserts that compilation really happens there,
so that coverage cannot silently fall back to eager.
"""

import pytest
import torch

from fme.ace.stepper.test_single_module import get_regression_stepper_and_data
from fme.core.optimization import OptimizationConfig
from fme.core.step.single_module import SingleModuleStep
from fme.core.testing import dynamo_hygiene  # noqa: F401  autouse in this module


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compilation defaults on only on CUDA"
)
def test_stepper_compiles_by_default_on_cuda():
    torch.manual_seed(0)
    train_stepper, data = get_regression_stepper_and_data()
    step = train_stepper._stepper._step_obj
    assert isinstance(step, SingleModuleStep)
    assert step.module.is_compiled
    optimization = OptimizationConfig(optimizer_type="Adam", lr=1e-4).build(
        train_stepper.modules, max_epochs=1
    )
    train_output = train_stepper.train_on_batch(data, optimization)
    for name, value in train_output.metrics.items():
        assert torch.isfinite(value).all(), name
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
