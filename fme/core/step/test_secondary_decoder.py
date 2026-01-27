import pytest
import torch

from fme.core.registry import ModuleSelector

from .secondary_decoder import SecondaryDecoder, SecondaryDecoderConfig


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
        assert isinstance(decoder.torch_modules, torch.nn.ModuleList)

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
