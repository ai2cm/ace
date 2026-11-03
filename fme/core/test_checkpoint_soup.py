import dataclasses
import datetime

import pytest
import torch

from fme.ace.stepper import Stepper, StepperConfig
from fme.ace.stepper.parameter_init import ParameterInitializationConfig
from fme.ace.stepper.single_module import load_weights_and_history
from fme.core.checkpoint_soup import average_state_dicts, main, soup_checkpoints
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.ema import EMA_CHECKPOINT_KEY, EMATracker, load_ema_params_if_available
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector


def test_average_of_two_state_dicts_is_midpoint():
    a = {"w": torch.randn(3, 4)}
    b = {"w": torch.randn(3, 4)}
    result = average_state_dicts([a, b])
    torch.testing.assert_close(result["w"], (a["w"] + b["w"]) / 2)


def test_weighted_average_normalizes_weights():
    a = {"w": torch.zeros(2)}
    b = {"w": torch.ones(2)}
    result = average_state_dicts([a, b], weights=[1.0, 3.0])
    torch.testing.assert_close(result["w"], torch.full((2,), 0.75))


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param([1.0], id="wrong_length"),
        pytest.param([1.0, -1.0], id="negative"),
        pytest.param([0.0, 0.0], id="zero_sum"),
    ],
)
def test_invalid_weights_raise(weights):
    state = {"w": torch.zeros(2)}
    with pytest.raises(ValueError, match="weights|Weights"):
        average_state_dicts([state, state], weights=weights)


@pytest.mark.parametrize(
    "a, b, match",
    [
        pytest.param({"w": torch.zeros(2)}, {"v": torch.zeros(2)}, "keys", id="keys"),
        pytest.param({"w": torch.zeros(2)}, {"w": torch.zeros(3)}, "shape", id="shape"),
        pytest.param({"w": 1}, {"w": 2}, "differs", id="non_tensor"),
    ],
)
def test_mismatched_inputs_raise(a, b, match):
    with pytest.raises(ValueError, match=match):
        average_state_dicts([a, b])


def test_non_floating_tensors_are_copied_from_first():
    a = {"step": torch.tensor(1), "mask": torch.tensor([True, False])}
    b = {"step": torch.tensor(4), "mask": torch.tensor([False, True])}
    result = average_state_dicts([a, b])
    assert result["step"].dtype == torch.int64
    assert result["step"].item() == 1
    assert torch.equal(result["mask"], a["mask"])


@pytest.mark.parametrize(
    "dtype, accumulate_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_dtype_preserved_and_accumulated_in_at_least_float32(dtype, accumulate_dtype):
    # with weights of 1/3, accumulating in a lower precision gives a different result
    torch.manual_seed(0)
    states = [{"w": torch.randn(64).to(dtype)} for _ in range(3)]
    result = average_state_dicts(states)
    assert result["w"].dtype == dtype
    expected = sum((1 / 3) * s["w"].to(accumulate_dtype) for s in states).to(dtype)
    assert torch.equal(result["w"], expected)


def _get_stepper(seed: int) -> Stepper:
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="SphericalFourierNeuralOperatorNet",
                        config={"num_layers": 2, "embed_dim": 3},
                    ),
                    in_names=["x"],
                    out_names=["x"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(means={"x": 0.5}, stds={"x": 2.0}),
                    ),
                ),
            ),
        ),
    )
    img_shape = (16, 32)
    dataset_info = DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(img_shape[0], device=get_device()),
            lon=torch.zeros(img_shape[1], device=get_device()),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7), bk=torch.arange(7)
        ),
        timestep=datetime.timedelta(hours=6),
    )
    torch.manual_seed(seed)
    return config.get_stepper(dataset_info=dataset_info)


def _save_checkpoint(stepper: Stepper, path: str, epoch: int):
    ema = EMATracker(stepper.modules, decay=0.9)
    torch.save(
        {
            "epoch": epoch,
            "stepper": stepper.get_state(),
            EMA_CHECKPOINT_KEY: ema.get_state(include_params=True),
            "optimization": {"dummy": torch.zeros(1)},
        },
        path,
    )


def _assert_parameters_equal(stepper: Stepper, expected: dict[str, torch.Tensor]):
    params = dict(stepper.modules.named_parameters())
    assert params.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(params[name].detach().cpu(), value)


def test_soup_checkpoints_round_trip(tmp_path, caplog):
    stepper_a = _get_stepper(seed=0)
    stepper_b = _get_stepper(seed=1)
    path_a, path_b = str(tmp_path / "a.tar"), str(tmp_path / "b.tar")
    _save_checkpoint(stepper_a, path_a, epoch=3)
    _save_checkpoint(stepper_b, path_b, epoch=4)
    params_b = dict(stepper_b.modules.named_parameters())
    expected = {
        name: 0.75 * param.detach().cpu() + 0.25 * params_b[name].detach().cpu()
        for name, param in stepper_a.modules.named_parameters()
    }
    output_path = str(tmp_path / "soup.tar")

    main([output_path, path_a, path_b, "--weights", "3", "1"])

    assert "raw, non-EMA weights are averaged" in caplog.text
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    assert checkpoint["soup"] == {"sources": [path_a, path_b], "weights": [0.75, 0.25]}
    assert checkpoint["epoch"] == 3
    assert "optimization" not in checkpoint
    from_state = Stepper.from_state(checkpoint["stepper"])
    _assert_parameters_equal(from_state, expected)
    # stored EMA weights are dropped, so they cannot replace the soup weights
    assert not load_ema_params_if_available(checkpoint, from_state.modules, output_path)

    initializer = ParameterInitializationConfig(weights_path=output_path).build(
        load_weights_and_history
    )
    fresh = _get_stepper(seed=2)
    initializer.apply_weights(list(fresh.modules))
    _assert_parameters_equal(fresh, expected)


def test_soup_checkpoints_rejects_different_configs(tmp_path):
    stepper_a = _get_stepper(seed=0)
    path_a, path_b = str(tmp_path / "a.tar"), str(tmp_path / "b.tar")
    _save_checkpoint(stepper_a, path_a, epoch=0)
    checkpoint = torch.load(path_a, map_location="cpu", weights_only=False)
    checkpoint["stepper"]["config"]["step"]["config"]["in_names"] = ["y"]
    torch.save(checkpoint, path_b)
    with pytest.raises(ValueError, match="different stepper config"):
        soup_checkpoints([path_a, path_b], str(tmp_path / "soup.tar"))
