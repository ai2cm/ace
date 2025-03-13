import unittest.mock

import pytest
import torch

from fme.core.multi_call import MultiCallConfig

from .multi_call import MultiCallStep
from .step import StepABC


@pytest.mark.parametrize("include_multi_call_in_loss", [True, False])
def test_multi_call(include_multi_call_in_loss: bool):
    output_names = ["b", "c"]
    multi_call_output_names = ["c"]

    def _step(input, next_step_input_data, *_):
        prediction = {k: input["CO2"].detach().clone() for k in output_names}
        return prediction

    wrapped_step = unittest.mock.Mock(spec=StepABC)
    wrapped_step.input_names = ["a", "b", "CO2"]
    wrapped_step.output_names = output_names
    wrapped_step.prognostic_names = ["a", "b"]
    wrapped_step.diagnostic_names = ["c", "d"]
    wrapped_step.loss_names = ["c", "d"]
    wrapped_step.step.side_effect = _step

    step = MultiCallStep(
        wrapped_step,
        MultiCallConfig(
            forcing_name="CO2",
            forcing_multipliers={"_doubled_co2": 2},
            output_names=multi_call_output_names,
        ),
        include_multi_call_in_loss=include_multi_call_in_loss,
    )

    assert step.output_names == ["b", "c", "c_doubled_co2"]
    if include_multi_call_in_loss:
        assert step.loss_names == ["c", "d", "c_doubled_co2"]
    else:
        assert step.loss_names == ["c", "d"]

    input = {
        "a": torch.randn(1, 2, 3, 4),
        "b": torch.randn(1, 2, 3, 4),
        "CO2": torch.randn(1, 2, 3, 4),
    }
    out = step.step(input, {})
    torch.testing.assert_close(out["b"], input["CO2"])
    torch.testing.assert_close(out["c"], input["CO2"])
    torch.testing.assert_close(out["c_doubled_co2"], input["CO2"] * 2)
