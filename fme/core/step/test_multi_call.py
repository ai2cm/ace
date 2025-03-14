import unittest.mock

import pytest
import torch

from fme.core.dataset_info import DatasetInfo
from fme.core.multi_call import MultiCallConfig

from .multi_call import MultiCallStepConfig
from .step import StepSelector
from .test_step_registry import MockStep


@pytest.mark.parametrize("include_multi_call_in_loss", [True, False])
def test_multi_call(include_multi_call_in_loss: bool):
    output_names = ["b", "c"]
    multi_call_output_names = ["c"]

    def _step(input, next_step_input_data, *_):
        prediction = {k: input["CO2"].detach().clone() for k in output_names}
        return prediction

    config = MultiCallStepConfig(
        wrapped_step=StepSelector(
            type="mock",
            config={"in_names": ["CO2", "a", "b"], "out_names": ["b", "c"]},
        ),
        config=MultiCallConfig(
            forcing_name="CO2",
            forcing_multipliers={"_doubled_co2": 2},
            output_names=multi_call_output_names,
        ),
        include_multi_call_in_loss=include_multi_call_in_loss,
    )

    with unittest.mock.patch.object(MockStep, "step", side_effect=_step):
        step = config.get_step(DatasetInfo())
        assert step.output_names == ["b", "c", "c_doubled_co2"]
        if include_multi_call_in_loss:
            assert step.loss_names == ["b", "c", "c_doubled_co2"]
        else:
            assert step.loss_names == ["b", "c"]

        input = {
            "a": torch.randn(1, 2, 3, 4),
            "b": torch.randn(1, 2, 3, 4),
            "CO2": torch.randn(1, 2, 3, 4),
        }
        out = step.step(input, {})
    torch.testing.assert_close(out["b"], input["CO2"])
    torch.testing.assert_close(out["c"], input["CO2"])
    torch.testing.assert_close(out["c_doubled_co2"], input["CO2"] * 2)
