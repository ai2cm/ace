import contextlib

import pytest
import torch
from torch import nn

from fme.core.wildcard import (
    ConflictingRuleError,
    MissingParameterError,
    UnusedRuleError,
    apply_by_wildcard,
    wildcard_match,
)


@pytest.mark.parametrize(
    "pattern, name, expected",
    [
        ("*", "abc", True),
        ("*", "abc.def", True),
        ("abc*", "abc", True),
        ("abc*", "abc.def", True),
        ("abc*", "def", False),
        ("abc.*", "abc.def", True),
        ("*.abc.*", "abc.def", False),
        ("*.def.*", "abc.def", False),
        ("*.def.*", "abc.def.ghi", True),
        ("abc.*", "abc.def.ghi", True),
        ("abc.*.ghi", "abc.def.ghi", True),
        ("abc.*.ghi", "abc.def", False),
        ("abc.*.ghi", "abc.def.ghi.jkl", False),
        ("*.abc.ghi", "def.abc.ghi", True),
        ("*.abc.ghi", "abc.ghi", False),
        ("*.abc.ghi", "def.abc.ghi.jkl", False),
    ],
)
def test_wildcard_match(pattern, name, expected):
    assert wildcard_match(pattern=pattern, name=name) == expected


class NestedModule2(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))


class NestedModule1(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))
        self.nested = NestedModule2()


@pytest.mark.parametrize(
    "include, exclude, expected_applied, raise_if_unused, expected_error",
    [
        pytest.param(
            ["*"], [], ["weight", "nested.weight"], True, None, id="include all"
        ),
        pytest.param([], ["*"], [], True, None, id="exclude all"),
        pytest.param(
            ["weight"], ["nested.*"], ["weight"], True, None, id="weight included"
        ),
        pytest.param(
            ["*"],
            ["nested.*"],
            [],
            True,
            ConflictingRuleError,
            id="nested_param_in_both",
        ),
        pytest.param(
            ["*"],
            ["weight"],
            [],
            True,
            ConflictingRuleError,
            id="star_include_with_an_exclude",
        ),
        pytest.param(
            [],
            ["weight"],
            [],
            True,
            MissingParameterError,
            id="missing_weight_using_exclude",
        ),
        pytest.param(
            ["weight"],
            [],
            [],
            True,
            MissingParameterError,
            id="missing_weight_using_include",
        ),
        pytest.param(
            ["*.weight"],
            [],
            [],
            True,
            MissingParameterError,
            id="missing_weight_using_wildcard_include",
        ),
        pytest.param(
            ["foo"],
            ["*"],
            [],
            True,
            UnusedRuleError,
            id="unused_include_rule",
        ),
        pytest.param(
            ["*"],
            ["foo"],
            [],
            True,
            UnusedRuleError,
            id="unused_exclude_rule",
        ),
        pytest.param(
            ["foo"],
            ["*"],
            [],
            False,
            None,
            id="unused_include_rule_allowed",
        ),
        pytest.param(
            ["*"],
            ["foo"],
            ["weight", "nested.weight"],
            False,
            None,
            id="unused_exclude_rule_allowed",
        ),
        pytest.param(
            ["weight", "foo"],
            ["*"],
            [],
            True,
            ConflictingRuleError,
            id="unused_and_conflicting_include_rule",
        ),
    ],
)
def test_apply_by_wildcard(
    include: list[str],
    exclude: list[str],
    expected_applied: list[str],
    raise_if_unused: bool,
    expected_error: type[Exception] | None,
):
    model = NestedModule1()

    def func(module: nn.Module, name: str):
        module.get_parameter(name).requires_grad = False

    if expected_error is not None:
        context = pytest.raises(expected_error)
    else:
        context = contextlib.nullcontext()

    with context:
        apply_by_wildcard(
            model,
            func,
            include,
            exclude,
            raise_if_unused=raise_if_unused,
        )

    if expected_error is None:
        for name, param in model.named_parameters():
            assert param.requires_grad == (name not in expected_applied)


def test_apply_by_wildcard_empty_wildcard_allowed():
    model = nn.Module()
    assert len(model.state_dict()) == 0
    apply_by_wildcard(model, lambda x, y: None, [], ["*"], raise_if_unused=True)
