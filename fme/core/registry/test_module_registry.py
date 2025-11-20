import dataclasses
from collections.abc import Iterable

import dacite
import pytest
import torch

from fme.core.dataset_info import DatasetInfo

from .module import ModuleConfig, ModuleSelector


class MockModule(torch.nn.Module):
    def __init__(self, param_shapes: Iterable[tuple[int, ...]]):
        super().__init__()
        for i, shape in enumerate(param_shapes):
            setattr(self, f"param{i}", torch.nn.Parameter(torch.randn(shape)))


@ModuleSelector.register("mock")
@dataclasses.dataclass
class MockModuleBuilder(ModuleConfig):
    param_shapes: list[tuple[int, ...]]

    def build(self, n_in_channels, n_out_channels, dataset_info):
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
    dataset_info = DatasetInfo(img_shape=(16, 32))
    module = selector.build(1, 1, dataset_info)
    assert isinstance(module, MockModule)


def test_module_selector_raises_with_bad_config():
    with pytest.raises(dacite.UnexpectedDataError):
        ModuleSelector(type="mock", config={"non_existent_key": 1})
