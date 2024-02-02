"""The registry also performs configuration set up so it needs to be tested."""
import copy
from typing import Mapping

import pytest
import torch

from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.device import get_device
from fme.core.normalizer import FromStateNormalizer
from fme.core.stepper import SingleModuleStepperConfig
from fme.fcn_training.registry import with_weights


def test_builder_with_weights_loads_same_state(tmpdir):
    sfno_config_data = {
        "type": "SphericalFourierNeuralOperatorNet",
        "config": {
            "num_layers": 2,
            "embed_dim": 3,
        },
    }
    stepper_config_data = {
        "builder": sfno_config_data,
        "in_names": ["x"],
        "out_names": ["x"],
        "normalization": FromStateNormalizer(
            state={
                "means": {"x": torch.randn(1)},
                "stds": {"x": torch.randn(1)},
            }
        ),
    }
    area = torch.ones((1, 16, 32)).to(get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7)).to(
        get_device()
    )
    stepper_config = SingleModuleStepperConfig.from_state(stepper_config_data)
    stepper = stepper_config.get_stepper(
        img_shape=(16, 32),
        area=area,
        sigma_coordinates=sigma_coordinates,
    )
    torch.save(
        {
            "stepper": stepper.get_state(),
        },
        str(tmpdir / "weights.ckpt"),
    )
    with_builder_config = {
        "type": "BuilderWithWeights",
        "config": {
            "module": sfno_config_data,
            "weights_path": str(tmpdir / "weights.ckpt"),
            "allow_missing_parameters": False,
        },
    }
    with_builder_stepper_config_data = {
        "builder": with_builder_config,
        "in_names": ["x"],
        "out_names": ["x"],
        "normalization": FromStateNormalizer(
            state={
                "means": {"x": torch.randn(1)},
                "stds": {"x": torch.randn(1)},
            }
        ),
    }
    with_builder_stepper = SingleModuleStepperConfig.from_state(
        with_builder_stepper_config_data
    ).get_stepper(
        img_shape=(16, 32),
        area=area,
        sigma_coordinates=sigma_coordinates,
    )
    assert_same_state(
        with_builder_stepper.module.state_dict(),
        stepper.module.state_dict(),
        allow_larger=False,
    )


def assert_same_state(
    state1: Mapping[str, torch.Tensor],
    state2: Mapping[str, torch.Tensor],
    allow_larger: bool,
    same_keys: bool = True,
):
    """
    Assert that two states are the same.

    If allow_larger is True then the states
    may have different dimension lengths, and if they do then the initial slice
    along that dimension will be compared.

    Args:
        state1: The first state.
        state2: The second state.
        allow_larger: Whether the states can have different dimension lengths.
        same_keys: Whether the states must have the same keys.
    """
    if same_keys:
        assert state1.keys() == state2.keys()
    same_state = []
    different_state = []
    for key in state1:
        if key in state2:
            if allow_larger:
                s = [
                    slice(0, min(s1, s2))
                    for s1, s2 in zip(state1[key].shape, state2[key].shape)
                ]
            else:
                s = [slice(0, None) for _ in state1[key].shape]
            if torch.allclose(state1[key][s], state2[key][s]):
                if not torch.all(state1[key] == 0):
                    same_state.append(key)
            else:
                different_state.append(key)
    assert len(same_state) > 0, (
        "No parameters were the same between the two states, "
        "or all parameters were uninitialized."
    )
    assert len(different_state) == 0, (
        f"Parameters {different_state} were different between "
        f"the two states, only {same_state} had the same non-zero state."
    )


@pytest.mark.parametrize(
    "loaded_shape,built_shape,extra_built_layer,expect_exception",
    [
        pytest.param((8, 16), (8, 16), False, False, id="same_shape"),
        pytest.param((8, 16), (8, 16), True, False, id="extra_layer"),
        pytest.param((8, 16), (16, 32), False, False, id="larger_model"),
        pytest.param((16, 32), (8, 16), False, True, id="smaller_model"),
    ],
)
def test_builder_with_weights_sfno_init(
    tmpdir, loaded_shape, built_shape, extra_built_layer: bool, expect_exception: bool
):
    sfno_config_data = {
        "type": "SphericalFourierNeuralOperatorNet",
        "config": {
            "num_layers": 2,
            "embed_dim": 3,
            "scale_factor": 1,
        },
    }
    stepper_config_data = {
        "builder": sfno_config_data,
        "in_names": ["x"],
        "out_names": ["x"],
        "normalization": FromStateNormalizer(
            state={
                "means": {"x": torch.randn(1)},
                "stds": {"x": torch.randn(1)},
            }
        ),
    }
    area = torch.ones((1, 16, 32)).to(get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7)).to(
        get_device()
    )
    stepper_config = SingleModuleStepperConfig.from_state(stepper_config_data)
    stepper = stepper_config.get_stepper(
        img_shape=loaded_shape,
        area=area,
        sigma_coordinates=sigma_coordinates,
    )
    built_sfno_config_data = copy.deepcopy(sfno_config_data)
    if extra_built_layer:
        built_sfno_config_data["config"]["num_layers"] += 1  # type: ignore
    torch.save(
        {
            "stepper": stepper.get_state(),
        },
        str(tmpdir / "weights.ckpt"),
    )
    with_builder_config = {
        "type": "BuilderWithWeights",
        "config": {
            "module": built_sfno_config_data,
            "weights_path": str(tmpdir / "weights.ckpt"),
            "allow_missing_parameters": False,
        },
    }
    with_builder_stepper_config_data = {
        "builder": with_builder_config,
        "in_names": ["x"],
        "out_names": ["x"],
        "normalization": FromStateNormalizer(
            state={
                "means": {"x": torch.randn(1)},
                "stds": {"x": torch.randn(1)},
            }
        ),
    }
    if expect_exception:
        with pytest.raises(ValueError):
            with_builder_stepper = SingleModuleStepperConfig.from_state(
                with_builder_stepper_config_data
            ).get_stepper(
                img_shape=built_shape,
                area=area,
                sigma_coordinates=sigma_coordinates,
            )
    else:
        with_builder_stepper = SingleModuleStepperConfig.from_state(
            with_builder_stepper_config_data
        ).get_stepper(
            img_shape=built_shape,
            area=area,
            sigma_coordinates=sigma_coordinates,
        )
        print(with_builder_stepper.module.state_dict().keys())
        print(stepper.module.state_dict().keys())
        if extra_built_layer:
            with pytest.raises(AssertionError):
                assert_same_state(
                    with_builder_stepper.module.state_dict(),
                    stepper.module.state_dict(),
                    allow_larger=True,
                    same_keys=True,  # This should fail if there are more parameters
                )
        assert_same_state(
            with_builder_stepper.module.state_dict(),
            stepper.module.state_dict(),
            allow_larger=True,
            same_keys=(not extra_built_layer),
        )


class SimpleLinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.custom_param = torch.nn.Parameter(torch.randn(5, 5))


class NestedModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = SimpleLinearModule(in_features, out_features)
        self.linear2 = SimpleLinearModule(out_features, in_features)
        self.custom_param = torch.nn.Parameter(torch.randn(5, 5))


@pytest.mark.parametrize(
    "from_module, to_module, expected_exception",
    [
        pytest.param(
            SimpleLinearModule(10, 10),
            SimpleLinearModule(10, 10),
            None,
            id="Matching sizes",
        ),
        pytest.param(
            SimpleLinearModule(10, 10),
            SimpleLinearModule(20, 10),
            None,
            id="to_module larger weights",
        ),
        pytest.param(
            SimpleLinearModule(20, 10),
            SimpleLinearModule(10, 10),
            ValueError,
            id="from_module larger weights",
        ),
        pytest.param(
            NestedModule(10, 20),
            NestedModule(10, 20),
            None,
            id="Complex modules matching sizes",
        ),
        pytest.param(
            NestedModule(10, 20),
            SimpleLinearModule(10, 20),
            ValueError,
            id="Nested modules, mismatched structure",
        ),
    ],
)
def test_overwrite_weights(from_module, to_module, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            with_weights._overwrite_weights(from_module, to_module)
    else:
        with_weights._overwrite_weights(from_module, to_module)
        for from_param, to_param in zip(
            from_module.parameters(), to_module.parameters()
        ):
            if len(from_param.shape) == 1:
                assert torch.allclose(
                    from_param.data, to_param.data[: from_param.data.size(0)]
                )
            else:
                assert torch.allclose(
                    from_param.data,
                    to_param.data[: from_param.data.size(0), : from_param.data.size(1)],
                )
