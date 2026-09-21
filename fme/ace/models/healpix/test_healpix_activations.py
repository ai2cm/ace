import pytest
import torch

from fme.ace.models.healpix.healpix_activations import CappedGELU


def _reference_forward(inputs: torch.Tensor, cap_value: float) -> torch.Tensor:
    """Previous implementation, which read the cap via ``.item()``."""
    x = torch.nn.GELU()(inputs)
    return torch.clamp(x, max=cap_value)


def _inputs_spanning_cap(cap_value: float) -> torch.Tensor:
    """Random inputs whose GELU outputs fall on both sides of the cap."""
    torch.manual_seed(0)
    return torch.randn(64, 3, 8, 16) * cap_value * 2.0


@pytest.mark.parametrize("cap_value", [1.0, 10.0])
def test_capped_gelu_matches_item_based_reference(cap_value: float):
    module = CappedGELU(cap_value=cap_value)
    inputs = _inputs_spanning_cap(cap_value)
    result = module(inputs)
    assert (result == cap_value).any(), "inputs should exercise the cap"
    assert (result < cap_value).any(), "inputs should exercise the uncapped path"
    reference = _reference_forward(inputs, cap_value)
    assert result.dtype == reference.dtype
    assert torch.equal(result, reference)


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
