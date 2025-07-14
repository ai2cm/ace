import dataclasses
import unittest.mock

import pytest
import torch

from fme.core.dataset_info import DatasetInfo
from fme.core.multi_call import MultiCallConfig
from fme.core.normalizer import StandardNormalizer

from .multi_call import (
    MultiCallStepConfig,
    _extend_normalizer_with_multi_call_outputs,
    replace_multi_call,
)
from .step import StepSelector
from .test_step_registry import MockStep, MockStepConfig


@pytest.mark.parametrize("include_multi_call_in_loss", [True, False])
def test_multi_call(include_multi_call_in_loss: bool):
    output_names = ["b", "c"]
    multi_call_output_names = ["c"]

    def _step(input, next_step_input_data, wrapper):
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
        step = config.get_step(DatasetInfo(), lambda x: None)
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


def test_replace_multi_call_adds_wrapper():
    selector = StepSelector(
        type="mock", config={"in_names": ["a", "b"], "out_names": ["c", "d"]}
    )
    multi_call = MultiCallConfig(
        forcing_name="a",
        forcing_multipliers={"_doubled_a": 2},
        output_names=["c"],
    )
    state = {"foo": "bar"}
    new_selector, new_state = replace_multi_call(selector, multi_call, state)
    assert new_selector.type == "multi_call"
    assert new_selector.config["wrapped_step"] == dataclasses.asdict(selector)
    assert new_selector.config["config"] == dataclasses.asdict(multi_call)
    new_step_config = new_selector._step_config_instance
    assert isinstance(new_step_config, MultiCallStepConfig)
    assert new_step_config.wrapped_step == selector
    assert new_step_config.config == multi_call
    assert new_step_config.include_multi_call_in_loss
    assert new_state["wrapped_step"] == state


def test_replace_multi_call_updates_wrapper():
    mock_selector = StepSelector(
        type="mock", config={"in_names": ["a", "b"], "out_names": ["c", "d"]}
    )
    multi_call = MultiCallConfig(
        forcing_name="a",
        forcing_multipliers={"_doubled_a": 2},
        output_names=["c"],
    )
    selector = StepSelector(
        type="multi_call",
        config={
            "wrapped_step": dataclasses.asdict(mock_selector),
            "config": dataclasses.asdict(multi_call),
        },
    )
    new_multi_call = MultiCallConfig(
        forcing_name="b",
        forcing_multipliers={"_doubled_b": 2},
        output_names=["d"],
    )
    state = {"wrapped_step": {"foo": "bar"}}
    new_selector, new_state = replace_multi_call(selector, new_multi_call, state)
    assert new_selector.type == "multi_call"
    assert new_selector.config["wrapped_step"] == dataclasses.asdict(mock_selector)
    assert new_selector.config["config"] == dataclasses.asdict(new_multi_call)
    new_step_config = new_selector._step_config_instance
    assert isinstance(new_step_config, MultiCallStepConfig)
    assert isinstance(new_step_config.wrapped_step, StepSelector)
    assert new_step_config.wrapped_step.type == "mock"
    assert new_step_config.wrapped_step == mock_selector
    assert new_step_config.config == new_multi_call
    assert new_state == state


def test_replace_multi_call_updates_wrapper_with_none():
    mock_selector = StepSelector(
        type="mock", config={"in_names": ["a", "b"], "out_names": ["c", "d"]}
    )
    multi_call = MultiCallConfig(
        forcing_name="a",
        forcing_multipliers={"_doubled_a": 2},
        output_names=["c"],
    )
    selector = StepSelector(
        type="multi_call",
        config={
            "wrapped_step": dataclasses.asdict(mock_selector),
            "config": dataclasses.asdict(multi_call),
        },
    )
    state = {"wrapped_step": {"foo": "bar"}}
    new_selector, new_state = replace_multi_call(selector, None, state)
    assert new_selector.type == "multi_call"
    assert new_selector.config["wrapped_step"] == dataclasses.asdict(mock_selector)
    assert new_selector.config["config"] is None
    new_step_config = new_selector._step_config_instance
    assert isinstance(new_step_config, MultiCallStepConfig)
    assert new_step_config.wrapped_step.type == "mock"
    assert new_step_config.wrapped_step == mock_selector
    assert new_step_config.config is None
    assert new_state == state


def test_extend_normalizer_with_multi_call_outputs():
    torch.manual_seed(0)
    c_mean, c_std = torch.randn(2)
    normalizer = StandardNormalizer(
        means={"a": torch.tensor(0), "b": torch.tensor(0), "c": c_mean},
        stds={"a": torch.tensor(1), "b": torch.tensor(1), "c": c_std},
    )
    config = MultiCallConfig(
        forcing_name="CO2",
        forcing_multipliers={"_doubled_co2": 2, "_halved_co2": 0.5},
        output_names=["c"],
    )
    result = _extend_normalizer_with_multi_call_outputs(config, normalizer)
    assert result.means == {
        "a": torch.tensor(0),
        "b": torch.tensor(0),
        "c": c_mean,
        "c_doubled_co2": c_mean,
        "c_halved_co2": c_mean,
    }
    assert result.stds == {
        "a": torch.tensor(1),
        "b": torch.tensor(1),
        "c": c_std,
        "c_doubled_co2": c_std,
        "c_halved_co2": c_std,
    }


@pytest.mark.parametrize("include_multi_call_in_loss", [True, False])
def test_loss_normalizer_uses_extra_stats_names(include_multi_call_in_loss: bool):
    torch.manual_seed(0)
    a_mean, a_std = torch.randn(2)
    b_mean, b_std = torch.randn(2)
    b_doubled_mean, b_doubled_std = torch.randn(2)
    b_halved_mean, b_halved_std = torch.randn(2)
    c_mean, c_std = torch.randn(2)
    d_mean, d_std = torch.randn(2)
    c_doubled_mean, c_doubled_std = torch.randn(2)
    c_halved_mean, c_halved_std = torch.randn(2)
    means = {
        "a": a_mean,
        "b": b_mean,
        "b_doubled_co2": b_doubled_mean,
        "b_halved_co2": b_halved_mean,
        "c": c_mean,
        "c_doubled_co2": c_doubled_mean,
        "c_halved_co2": c_halved_mean,
        "d": d_mean,
    }
    stds = {
        "a": a_std,
        "b": b_std,
        "b_doubled_co2": b_doubled_std,
        "b_halved_co2": b_halved_std,
        "c": c_std,
        "c_doubled_co2": c_doubled_std,
        "c_halved_co2": c_halved_std,
        "d": d_std,
    }

    def _get_loss_normalizer(
        extra_names: list[str] | None,
        extra_residual_scaled_names: list[str] | None,
    ):
        base_names = ["a", "b", "c", "d"]
        if extra_names is None:
            extra_names = []
        if extra_residual_scaled_names is None:
            extra_residual_scaled_names = []
        extra_names = extra_names + extra_residual_scaled_names
        return StandardNormalizer(
            means={
                **{k: means[k] for k in base_names},
                **{k: means[k] for k in extra_names},
            },
            stds={
                **{k: stds[k] for k in base_names},
                **{k: stds[k] for k in extra_names},
            },
        )

    mock_get_loss_normalizer = unittest.mock.Mock(side_effect=_get_loss_normalizer)
    with unittest.mock.patch.object(
        MockStepConfig, "get_loss_normalizer", side_effect=mock_get_loss_normalizer
    ):
        config = MultiCallStepConfig(
            wrapped_step=StepSelector(
                type="mock",
                config={"in_names": ["CO2", "a", "b"], "out_names": ["b", "c", "d"]},
            ),
            config=MultiCallConfig(
                forcing_name="CO2",
                forcing_multipliers={"_doubled_co2": 2, "_halved_co2": 0.5},
                output_names=["b", "c"],
            ),
            # we expect multi-call in loss normalizer regardless of
            # include_multi_call_in_loss
            include_multi_call_in_loss=include_multi_call_in_loss,
        )
        normalizer = config.get_loss_normalizer()
        assert normalizer.means == {
            "a": a_mean,
            "b": b_mean,
            "b_doubled_co2": b_doubled_mean,
            "b_halved_co2": b_halved_mean,
            "c": c_mean,
            "c_doubled_co2": c_doubled_mean,
            "c_halved_co2": c_halved_mean,
            "d": d_mean,
        }
        assert normalizer.stds == {
            "a": a_std,
            "b": b_std,
            "b_doubled_co2": b_doubled_std,
            "b_halved_co2": b_halved_std,
            "c": c_std,
            "c_doubled_co2": c_doubled_std,
            "c_halved_co2": c_halved_std,
            "d": d_std,
        }
        assert mock_get_loss_normalizer.call_count == 1
        call_args = mock_get_loss_normalizer.call_args
        assert set(call_args.kwargs["extra_names"]) == {
            "b_doubled_co2",
            "b_halved_co2",
            "c_doubled_co2",
            "c_halved_co2",
        }
        assert set(call_args.kwargs["extra_residual_scaled_names"]) == {
            "b_doubled_co2",
            "b_halved_co2",
        }
