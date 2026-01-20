import pytest
import torch

from fme.core.dataset_info import DatasetInfo
from fme.core.registry import ModuleSelector

from .secondary_decoder import MLPConfig, SecondaryDecoder, SecondaryDecoderConfig


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


class TestSecondaryDecoderConfig:
    def test_valid_mlp_network_type(self):
        # Should not raise
        config = SecondaryDecoderConfig(
            secondary_diagnostic_names=["diag1", "diag2"],
            network=ModuleSelector(type="MLP", config={}),
        )
        assert config.secondary_diagnostic_names == ["diag1", "diag2"]

    def test_invalid_network_type_raises_error(self):
        with pytest.raises(ValueError, match="Invalid network type"):
            SecondaryDecoderConfig(
                secondary_diagnostic_names=["diag1"],
                network=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet", config={}
                ),
            )


class TestSecondaryDecoder:
    def test_forward_and_unpack_integration(self):
        network = ModuleSelector(type="MLP", config={"hidden_dim": 16, "depth": 2})
        decoder = SecondaryDecoder(
            in_dim=4,
            out_names=["diag1", "diag2"],
            network=network,
        )
        # Input: [batch, channels, height, width]
        x = torch.randn(2, 4, 8, 8)
        output = decoder(x)
        assert isinstance(output, dict)
        assert set(output.keys()) == {"diag1", "diag2"}
        assert output["diag1"].shape == (2, 8, 8)
        assert output["diag2"].shape == (2, 8, 8)

    def test_module_property_returns_nn_module(self):
        network = ModuleSelector(type="MLP", config={})
        decoder = SecondaryDecoder(
            in_dim=4,
            out_names=["diag1"],
            network=network,
        )
        assert isinstance(decoder.module.torch_module, torch.nn.Module)

    def test_module_state_dict_and_load_module_state_dict(self):
        network = ModuleSelector(type="MLP", config={"hidden_dim": 16, "depth": 2})
        decoder1 = SecondaryDecoder(
            in_dim=4,
            out_names=["diag1", "diag2"],
            network=network,
        )
        decoder2 = SecondaryDecoder(
            in_dim=4,
            out_names=["diag1", "diag2"],
            network=network,
        )
        # Save state from decoder1, load into decoder2
        state = decoder1.get_module_state()
        decoder2.load_module_state(state)
        # Check that parameters match
        x = torch.randn(1, 4, 4, 4)
        torch.testing.assert_close(decoder1(x), decoder2(x))

    def test_to_method(self):
        network = ModuleSelector(type="MLP", config={})
        decoder = SecondaryDecoder(
            in_dim=4,
            out_names=["diag1"],
            network=network,
        )
        # Just test that to() returns self and doesn't error
        result = decoder.to("cpu")
        assert result is decoder
