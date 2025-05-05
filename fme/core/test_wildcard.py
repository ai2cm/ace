import contextlib

import pytest
import torch
from torch import nn

from fme.core.wildcard import apply_by_wildcard, wildcard_match


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
    "include, exclude, expected_applied, expected_error",
    [
        pytest.param(["*"], [], ["weight", "nested.weight"], None, id="include all"),
        pytest.param([], ["*"], [], None, id="exclude all"),
        pytest.param(["weight"], ["nested.*"], ["weight"], None, id="weight included"),
        pytest.param(["*"], ["nested.*"], [], ValueError, id="nested param in both"),
        pytest.param(["*"], ["weight"], [], ValueError, id="* include with an exclude"),
        pytest.param([], ["weight"], [], ValueError, id="missing weight using exclude"),
        pytest.param(["weight"], [], [], ValueError, id="missing weight using include"),
        pytest.param(
            ["*.weight"], [], [], ValueError, id="mising weight using wildcard include"
        ),
    ],
)
def test_apply_by_wildcard(
    include: list[str],
    exclude: list[str],
    expected_applied: list[str],
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
        )

    if expected_error is None:
        for name, param in model.named_parameters():
            assert param.requires_grad == (name not in expected_applied)
