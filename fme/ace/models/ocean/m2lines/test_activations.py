import pytest
import torch

from fme.ace.models.ocean.m2lines.activations import CappedGELU


def _reference_forward(inputs: torch.Tensor, cap_value: float) -> torch.Tensor:
    """Previous implementation, which read the cap via ``.item()``."""
    x = torch.nn.GELU()(inputs)
    return torch.clamp(x, max=cap_value)


def _inputs_spanning_cap(cap_value: float) -> torch.Tensor:
    """Random inputs whose GELU outputs fall on both sides of the cap."""
    torch.manual_seed(0)
    return torch.randn(64, 3, 8, 16) * cap_value * 2.0


def _forward_and_grad(fn, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = inputs.clone().requires_grad_(True)
    output = fn(inputs)
    output.sum().backward()
    assert inputs.grad is not None
    return output, inputs.grad


def _assert_matches_reference(
    module: CappedGELU, inputs: torch.Tensor, cap_value: float
):
    result, grad = _forward_and_grad(module, inputs)
    assert (result == cap_value).any(), "inputs should exercise the cap"
    assert (result < cap_value).any(), "inputs should exercise the uncapped path"
    reference, reference_grad = _forward_and_grad(
        lambda x: _reference_forward(x, cap_value), inputs
    )
    assert result.dtype == reference.dtype
    assert torch.equal(result, reference)
    assert torch.equal(grad, reference_grad)


@pytest.mark.parametrize("cap_value", [1.0, 10.0])
def test_capped_gelu_matches_item_based_reference(cap_value: float):
    module = CappedGELU(cap_value=cap_value)
    _assert_matches_reference(module, _inputs_spanning_cap(cap_value), cap_value)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cap_value", [0.3, 10.0])
def test_capped_gelu_reduced_precision_matches_item_based_reference(
    dtype: torch.dtype, cap_value: float
):
    """clamp casts the float32 cap buffer to the input dtype, so fp16/bf16
    outputs and gradients match the ``.item()`` reference exactly, including
    for a cap (0.3) that is not representable in reduced precision."""
    module = CappedGELU(cap_value=cap_value)
    inputs = _inputs_spanning_cap(cap_value).to(dtype)
    _assert_matches_reference(module, inputs, cap_value)


def test_capped_gelu_under_autocast_matches_item_based_reference():
    module = CappedGELU(cap_value=1.0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _assert_matches_reference(module, _inputs_spanning_cap(1.0), 1.0)


def test_capped_gelu_has_no_graph_breaks():
    # dynamo's compile cache is process-global, so clear it to make sure this
    # test traces the module itself rather than reusing another test's result.
    torch._dynamo.reset()
    try:
        module = CappedGELU(cap_value=10.0)
        inputs = torch.randn(4, 3, 8, 16)
        explanation = torch._dynamo.explain(module)(inputs)
    finally:
        torch._dynamo.reset()
    assert explanation.graph_count == 1, "forward should be fully captured"
    assert explanation.graph_break_count == 0, explanation.break_reasons


def test_capped_gelu_state_dict_keys_unchanged():
    assert set(CappedGELU(cap_value=10.0).state_dict().keys()) == {"cap"}
