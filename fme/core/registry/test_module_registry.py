import dataclasses
from collections.abc import Iterable

import dacite
import pytest
import torch

from fme.core.labels import LabelEncoder
from fme.core.registry.module import Module

from .module import CONDITIONAL_BUILDERS, ModuleConfig, ModuleSelector


class MockModule(torch.nn.Module):
    def __init__(self, param_shapes: Iterable[tuple[int, ...]]):
        super().__init__()
        for i, shape in enumerate(param_shapes):
            setattr(self, f"param{i}", torch.nn.Parameter(torch.randn(shape)))


@ModuleSelector.register("mock")
@dataclasses.dataclass
class MockModuleBuilder(ModuleConfig):
    param_shapes: list[tuple[int, ...]]

    def build(self, n_in_channels, n_out_channels, n_labels, img_shape):
        return MockModule(self.param_shapes)

    @classmethod
    def from_state(cls, state):
        return cls(state["param_shapes"])

    def get_state(self):
        return {
            "param_shapes": self.param_shapes,
        }


def test_register():
    """Make sure that the registry is working as expected."""
    selector = ModuleSelector(type="mock", config={"param_shapes": [(1, 2, 3)]})
    module = selector.build(
        n_in_channels=1, n_out_channels=1, all_labels=set(), img_shape=(16, 32)
    )
    assert isinstance(module, Module)
    assert isinstance(module.torch_module, MockModule)
    assert module._label_encoder is None


def test_build_conditional():
    """Make sure that the registry is working as expected."""
    try:
        CONDITIONAL_BUILDERS.append("mock")
        selector = ModuleSelector(
            type="mock", conditional=True, config={"param_shapes": [(1, 2, 3)]}
        )
        module = selector.build(
            n_in_channels=1, n_out_channels=1, all_labels={"a", "b"}, img_shape=(16, 32)
        )
        assert isinstance(module, Module)
        assert isinstance(module.torch_module, MockModule)
        assert isinstance(module._label_encoder, LabelEncoder)
    finally:
        CONDITIONAL_BUILDERS.remove("mock")


def test_module_selector_raises_with_bad_config():
    with pytest.raises(dacite.UnexpectedDataError):
        ModuleSelector(type="mock", config={"non_existent_key": 1})
