import dataclasses
from typing import Iterable, List, Tuple

import torch

from .module import ModuleConfig, ModuleSelector


class MockModule(torch.nn.Module):
    def __init__(self, param_shapes: Iterable[Tuple[int, ...]]):
        super().__init__()
        for i, shape in enumerate(param_shapes):
            setattr(self, f"param{i}", torch.nn.Parameter(torch.randn(shape)))


@ModuleSelector.register("mock")
@dataclasses.dataclass
class MockModuleBuilder(ModuleConfig):
    param_shapes: List[Tuple[int, ...]]

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
    selector = ModuleSelector(type="mock", config={"param_shapes": [(1, 2, 3)]})
    module = selector.build(1, 1, (16, 32))
    assert isinstance(module, MockModule)
