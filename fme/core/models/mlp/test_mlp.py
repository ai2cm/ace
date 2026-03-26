import pytest
import torch

from fme.core.dataset_info import DatasetInfo
from fme.core.registry import ModuleSelector

from .mlp import MLPConfig


class TestMLPConfig:
    def test_depth_0_raises_error(self):
        config = MLPConfig(hidden_dim=8, depth=0)
        with pytest.raises(ValueError, match="depth must be >= 1"):
            config.build(n_in_channels=4, n_out_channels=2, dataset_info=DatasetInfo())

    def test_build_creates_mlp_single_layer(self):
        config = MLPConfig(hidden_dim=8, depth=1)
        module = config.build(
            n_in_channels=4, n_out_channels=2, dataset_info=DatasetInfo()
        )
        # depth=1: just one Conv2d layer
        assert len(module) == 1
        assert isinstance(module[0], torch.nn.Conv2d)
        assert module[0].in_channels == 4
        assert module[0].out_channels == 2

    def test_build_creates_mlp_two_layers(self):
        config = MLPConfig(hidden_dim=16, depth=2)
        module = config.build(
            n_in_channels=4, n_out_channels=2, dataset_info=DatasetInfo()
        )
        assert isinstance(module, torch.nn.Sequential)
        # depth=2: Conv2d, GELU, Conv2d
        assert len(module) == 3

    def test_default_values(self):
        config = MLPConfig()
        assert config.hidden_dim == 256
        assert config.depth == 2

    def test_registered_with_module_selector(self):
        selector = ModuleSelector(type="MLP", config={"hidden_dim": 32, "depth": 3})
        module = selector.build(
            n_in_channels=4, n_out_channels=2, dataset_info=DatasetInfo()
        )
        assert isinstance(module.torch_module, torch.nn.Module)
        # depth=3: Conv2d, GELU, Conv2d, GELU, Conv2d
        assert len(module.torch_module) == 5
