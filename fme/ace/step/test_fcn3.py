import pytest
import torch

import fme
from fme.ace.step.fcn3 import FCN3Config, FCN3Selector, FCN3Step, FCN3StepConfig
from fme.core.distributed.non_distributed import DummyWrapper
from fme.core.optimization import Checkpoint
from fme.core.step.args import StepArgs
from fme.core.testing import (
    compile_backend,
    dynamo_hygiene,  # noqa: F401  autouse in this module
    get_dataset_info,
    trivial_network_and_loss_normalization,
)
from fme.core.typing_ import TensorDict

IMG_SHAPE = (16, 32)
FORCING_NAMES = ["DSWRFtoa"]
ATMOSPHERE_PROGNOSTIC_NAMES = ["air_temperature"]
ATMOSPHERE_DIAGNOSTIC_NAMES = ["radiative_heating"]
ATMOSPHERE_LEVELS = 2
SURFACE_PROGNOSTIC_NAMES = ["PRESsfc"]
SURFACE_DIAGNOSTIC_NAMES = ["PRATEsfc"]


def _get_fcn3_step_config(
    compile: bool = False,
    normalization_layer: str = "none",
) -> FCN3StepConfig:
    """A minimal FCN3 step config, kept small so compiling it stays cheap."""
    packed_names = [
        f"{name}_{i}"
        for i in range(ATMOSPHERE_LEVELS)
        for name in ATMOSPHERE_PROGNOSTIC_NAMES + ATMOSPHERE_DIAGNOSTIC_NAMES
    ]
    return FCN3StepConfig(
        builder=FCN3Selector(
            type="FCN3",
            config=FCN3Config(
                scale_factor=1,
                atmo_embed_dim=2,
                surf_embed_dim=2,
                aux_embed_dim=2,
                num_layers=2,
                normalization_layer=normalization_layer,
            ),
        ),
        forcing_names=FORCING_NAMES,
        atmosphere_prognostic_names=ATMOSPHERE_PROGNOSTIC_NAMES,
        atmosphere_diagnostic_names=ATMOSPHERE_DIAGNOSTIC_NAMES,
        atmosphere_levels=ATMOSPHERE_LEVELS,
        surface_prognostic_names=SURFACE_PROGNOSTIC_NAMES,
        surface_diagnostic_names=SURFACE_DIAGNOSTIC_NAMES,
        normalization=trivial_network_and_loss_normalization(
            sorted(
                set(FORCING_NAMES)
                .union(packed_names)
                .union(SURFACE_PROGNOSTIC_NAMES)
                .union(SURFACE_DIAGNOSTIC_NAMES)
            )
        ),
        compile=compile,
    )


def _get_step(config: FCN3StepConfig) -> FCN3Step:
    dataset_info = get_dataset_info(img_shape=IMG_SHAPE, device=fme.get_device())
    return config.get_step(dataset_info, lambda _: None)


def _get_args(step: FCN3Step, n_samples: int = 2) -> StepArgs:
    def tensor_dict(names) -> TensorDict:
        return {
            name: torch.rand(n_samples, *IMG_SHAPE, device=fme.get_device())
            for name in names
        }

    return StepArgs(
        input=tensor_dict(step.input_names),
        next_step_input_data=tensor_dict(step.next_step_input_names),
        labels=None,
    )


def _grads(step: FCN3Step) -> TensorDict:
    return {
        name: param.grad
        for name, param in step.module.named_parameters()
        if param.grad is not None
    }


@pytest.mark.medium_duration
def test_fcn3_step_compile_flag():
    """compile=True runs the forward through torch.compile while leaving the
    checkpoint state identical in structure to the uncompiled step.

    The step config only exposes ``compile: bool``, so the backend is forced
    to ``aot_eager`` from the outside to keep the test within the suite's
    timeouts while still exercising dynamo tracing.
    """
    torch.manual_seed(0)
    eager_step = _get_step(_get_fcn3_step_config())
    with compile_backend("aot_eager"):
        compiled_step = _get_step(_get_fcn3_step_config(compile=True))
    eager_state = eager_step.get_state()
    compiled_step.load_state(eager_state)
    assert compiled_step.get_state()["module"].keys() == eager_state["module"].keys()

    args = _get_args(eager_step)
    eager_step.eval()
    compiled_step.eval()
    with compile_backend("aot_eager"), torch.no_grad():
        eager_out = eager_step.step(args).output
        compiled_out = compiled_step.step(args).output
    for name in eager_out:
        torch.testing.assert_close(
            compiled_out[name], eager_out[name], atol=1e-4, rtol=1e-4
        )


@pytest.mark.medium_duration
@pytest.mark.parametrize(
    "normalization_layer",
    [
        "layer_norm",
        # instance_norm_s2 is the only single-rank normalization which routes
        # through the vendored ``@torch.compile``d normalization kernels, so it
        # exercises a compiled region nested inside the compiled step.
        "instance_norm_s2",
    ],
)
def test_fcn3_step_compile_flag_train_gradients(normalization_layer: str):
    """A compiled step trains: the activation-checkpointed forward and its
    backward produce finite gradients matching the uncompiled step's.
    """
    torch.manual_seed(0)
    eager_step = _get_step(
        _get_fcn3_step_config(normalization_layer=normalization_layer)
    )
    with compile_backend("aot_eager"):
        compiled_step = _get_step(
            _get_fcn3_step_config(compile=True, normalization_layer=normalization_layer)
        )
    compiled_step.load_state(eager_step.get_state())
    args = _get_args(eager_step)
    eager_step.train()
    compiled_step.train()

    eager_out = eager_step.step(args, wrapper=Checkpoint({})).output
    sum(value.sum() for value in eager_out.values()).backward()
    with compile_backend("aot_eager"):
        compiled_out = compiled_step.step(args, wrapper=Checkpoint({})).output
        sum(value.sum() for value in compiled_out.values()).backward()

    eager_grads = _grads(eager_step)
    compiled_grads = _grads(compiled_step)
    assert len(eager_grads) > 0
    assert compiled_grads.keys() == eager_grads.keys()
    for name, grad in compiled_grads.items():
        assert torch.isfinite(grad).all(), name
        torch.testing.assert_close(grad, eager_grads[name], atol=1e-4, rtol=1e-4)


@pytest.mark.medium_duration
def test_fcn3_step_compiles_after_distributed_wrapping():
    """The compiled module wraps the distributed-wrapped module, not the bare
    one, so distributed data parallel is inside the compiled graph.
    """
    with compile_backend("aot_eager"):
        step = _get_step(_get_fcn3_step_config(compile=True))
    # the presence of the (single-rank) distributed wrapper inside the compiled
    # module is the contract under test, so its type is asserted directly
    assert isinstance(step.module, DummyWrapper)
    assert step._forward_module._orig_mod is step.module
