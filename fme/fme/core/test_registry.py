import dataclasses
from typing import Iterable, Tuple

import pytest
import torch

from fme.core import registry


class MockModule(torch.nn.Module):
    def __init__(self, param_shapes: Iterable[Tuple[int, ...]]):
        super().__init__()
        for i, shape in enumerate(param_shapes):
            setattr(self, f"param{i}", torch.nn.Parameter(torch.randn(shape)))


@dataclasses.dataclass
class MockModuleBuilder(registry.ModuleConfig):
    param_shapes: Iterable[Tuple[int, ...]]

    def build(self, n_in_channels, n_out_channels, img_shape):
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
    original_registry = registry.NET_REGISTRY
    try:
        registry.NET_REGISTRY = {}
        with pytest.raises(ValueError):
            selector = registry.ModuleSelector(
                type="mock",
                config={"param_shapes": [(1, 2, 3)]},
            )
        registry.register("mock")(MockModuleBuilder)
        selector = registry.ModuleSelector(
            type="mock",
            config={"param_shapes": [(1, 2, 3)]},
        )
        module = selector.build(1, 1, (16, 32))
        assert isinstance(module, MockModule)
    finally:
        registry.NET_REGISTRY = original_registry
