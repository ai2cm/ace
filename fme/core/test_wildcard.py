import contextlib

import pytest
import torch
from torch import nn

from fme.core.wildcard import (
    UnusedRuleError,
    apply_by_exclude,
    apply_by_include,
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
    "include, expected_applied, expected_error",
    [
        pytest.param(["*"], ["weight", "nested.weight"], None, id="include all"),
        pytest.param(["weight"], ["weight"], None, id="weight included"),
        pytest.param(
            ["nested.*"], ["nested.weight"], None, id="nested weight included"
        ),
        pytest.param(["missing.*"], [], UnusedRuleError, id="nested weight included"),
    ],
)
def test_apply_by_include(include, expected_applied, expected_error):
    model = NestedModule1()

    applied = set()

    def func(module: nn.Module, name: str):
        assert name not in applied
        applied.add(name)

    if expected_error is not None:
        context = pytest.raises(expected_error)
    else:
        context = contextlib.nullcontext()

    with context:
        apply_by_include(
            model,
            func,
            include,
        )

    if expected_error is None:
        assert applied == set(expected_applied)


@pytest.mark.parametrize(
    "exclude, expected_applied, expected_error",
    [
        pytest.param(["*"], [], None, id="exclude all"),
        pytest.param(["weight"], ["nested.weight"], None, id="weight excluded"),
        pytest.param(["nested.*"], ["weight"], None, id="nested weight excluded"),
        pytest.param(
            ["missing.*"],
            [],
            UnusedRuleError,
            id="unused_exclude_rule",
        ),
    ],
)
def test_apply_by_exclude(exclude, expected_applied, expected_error):
    model = NestedModule1()

    applied = set()

    def func(module: nn.Module, name: str):
        assert name not in applied
        applied.add(name)

    if expected_error is not None:
        context = pytest.raises(expected_error)
    else:
        context = contextlib.nullcontext()

    with context:
        apply_by_exclude(
            model,
            func,
            exclude,
        )

    if expected_error is None:
        assert applied == set(expected_applied)
