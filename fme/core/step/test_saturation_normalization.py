import pytest
import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.device import move_tensordict_to_device
from fme.core.humidity import saturation_specific_humidity
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.global_mean_removal import SharedGlobalMeanRemovalConfig
from fme.core.step.saturation_normalization import (
    SaturationNormalizationConfig,
    _relative_humidity_name,
    build_saturation_normalization,
)
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.testing.dataset_info import get_dataset_info

# Two-layer hybrid sigma-pressure coordinate: interface pressures are
# [0, 0.5 * ps, ps], so layer 0 sits at 0.25 * ps and layer 1 at 0.75 * ps.
_AK = torch.tensor([0.0, 0.0, 0.0])
_BK = torch.tensor([0.0, 0.5, 1.0])

_RH1 = _relative_humidity_name(1)


def _vertical_coordinate() -> HybridSigmaPressureCoordinate:
    return HybridSigmaPressureCoordinate(ak=_AK.clone(), bk=_BK.clone())


def _build_transform(configs, in_names, out_names):
    return build_saturation_normalization(
        configs, _vertical_coordinate(), in_names, out_names
    )


def _expected_qsat(temperature: float, surface_pressure: float, level: int):
    layer_pressure = 0.5 * (
        (_AK[level] + _BK[level] * surface_pressure)
        + (_AK[level + 1] + _BK[level + 1] * surface_pressure)
    )
    return saturation_specific_humidity(torch.tensor(temperature), layer_pressure)


# ── transform behavior ──────────────────────────────────────────────────


def test_forward_replace_writes_derived_rh_and_round_trips():
    names = ["specific_total_water_1", "air_temperature_1", "PRESsfc"]
    config = SaturationNormalizationConfig(
        names=["specific_total_water_1"], prediction=True, input="replace"
    )
    transform = _build_transform([config], names, names)
    q, temperature, ps = 0.006, 290.0, 1.0e5
    raw = move_tensordict_to_device(
        {
            "specific_total_water_1": torch.full((2, 4, 4), q),
            "air_temperature_1": torch.full((2, 4, 4), temperature),
            "PRESsfc": torch.full((2, 4, 4), ps),
        }
    )
    result, state = transform.forward_transform(raw, raw)
    qsat = _expected_qsat(temperature, ps, level=1)
    # the source q channel is replaced by the derived RH channel
    assert "specific_total_water_1" not in result
    torch.testing.assert_close(
        result[_RH1].cpu(), torch.full((2, 4, 4), (q / qsat).item())
    )
    torch.testing.assert_close(result["air_temperature_1"], raw["air_temperature_1"])
    # the network predicts the derived channel; inverse maps it back to q
    restored = transform.inverse_transform(result, state)
    assert _RH1 not in restored
    torch.testing.assert_close(
        restored["specific_total_water_1"], raw["specific_total_water_1"]
    )


def test_forward_append_adds_derived_rh_and_keeps_q():
    names = ["specific_total_water_1", "air_temperature_1", "PRESsfc"]
    config = SaturationNormalizationConfig(
        names=["specific_total_water_1"], prediction=False, input="append"
    )
    transform = _build_transform([config], names, names)
    assert transform.derived_names == [_RH1]
    q, temperature, ps = 0.006, 290.0, 1.0e5
    raw = move_tensordict_to_device(
        {
            "specific_total_water_1": torch.full((1, 4, 4), q),
            "air_temperature_1": torch.full((1, 4, 4), temperature),
            "PRESsfc": torch.full((1, 4, 4), ps),
        }
    )
    result, state = transform.forward_transform(raw, raw)
    # raw q kept, derived RH added alongside it
    torch.testing.assert_close(
        result["specific_total_water_1"], raw["specific_total_water_1"]
    )
    qsat = _expected_qsat(temperature, ps, level=1)
    torch.testing.assert_close(
        result[_RH1].cpu(), torch.full((1, 4, 4), (q / qsat).item())
    )
    # append does not predict RH, so inverse is a no-op
    restored = transform.inverse_transform(result, state)
    torch.testing.assert_close(
        restored["specific_total_water_1"], raw["specific_total_water_1"]
    )


def test_qsat_is_per_level():
    names = [
        "specific_total_water_0",
        "specific_total_water_1",
        "air_temperature_0",
        "air_temperature_1",
        "PRESsfc",
    ]
    config = SaturationNormalizationConfig(
        names=["specific_total_water_*"], prediction=True, input="replace"
    )
    transform = _build_transform([config], names, names)
    ps = 1.0e5
    raw = move_tensordict_to_device(
        {
            "specific_total_water_0": torch.full((1, 2, 2), 0.001),
            "specific_total_water_1": torch.full((1, 2, 2), 0.006),
            "air_temperature_0": torch.full((1, 2, 2), 240.0),
            "air_temperature_1": torch.full((1, 2, 2), 290.0),
            "PRESsfc": torch.full((1, 2, 2), ps),
        }
    )
    result, _ = transform.forward_transform(raw, raw)
    qsat0 = _expected_qsat(240.0, ps, level=0)
    qsat1 = _expected_qsat(290.0, ps, level=1)
    torch.testing.assert_close(
        result[_relative_humidity_name(0)].cpu(),
        torch.full((1, 2, 2), (0.001 / qsat0).item()),
    )
    torch.testing.assert_close(
        result[_relative_humidity_name(1)].cpu(),
        torch.full((1, 2, 2), (0.006 / qsat1).item()),
    )


def test_packer_and_residual_name_mapping():
    in_names = ["specific_total_water_0", "specific_total_water_1", "T_0", "T_1", "PS"]
    out_names = ["specific_total_water_0", "specific_total_water_1"]
    configs = [
        SaturationNormalizationConfig(
            names=["specific_total_water_0"], prediction=True, input="replace"
        ),
        SaturationNormalizationConfig(
            names=["specific_total_water_1"], prediction=False, input="append"
        ),
    ]
    transform = _build_transform(configs, in_names, out_names)
    rh0, rh1 = _relative_humidity_name(0), _relative_humidity_name(1)
    # replace swaps q->rh in place; append adds rh after the inputs
    assert transform.input_packer_names(in_names) == [
        rh0,
        "specific_total_water_1",
        "T_0",
        "T_1",
        "PS",
        rh1,
    ]
    # only the predicted field is swapped on the output side
    assert transform.output_packer_names(out_names) == [
        rh0,
        "specific_total_water_1",
    ]
    assert set(transform.derived_names) == {rh0, rh1}
    assert transform.residual_names(out_names) == [rh0, "specific_total_water_1"]


def test_build_returns_none_without_config():
    assert build_saturation_normalization(None, _vertical_coordinate(), [], []) is None
    assert build_saturation_normalization([], _vertical_coordinate(), [], []) is None


def test_build_requires_vertical_coordinate():
    config = SaturationNormalizationConfig(
        names=["specific_total_water_0"], input="append"
    )
    names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    with pytest.raises(ValueError, match="vertical coordinate"):
        build_saturation_normalization([config], None, names, names)


def test_build_rejects_level_outside_model():
    config = SaturationNormalizationConfig(
        names=["specific_total_water_5"], input="append"
    )
    names = ["specific_total_water_5", "air_temperature_5", "PRESsfc"]
    with pytest.raises(ValueError, match="outside the 2 model layers"):
        _build_transform([config], names, names)


# ── config validation ───────────────────────────────────────────────────


def _step_config(
    saturation,
    in_names,
    out_names,
    global_mean_removal=None,
    residual_prediction=False,
):
    all_names = sorted(set(in_names) | set(out_names))
    return SingleModuleStepConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": torch.nn.Identity()}),
        in_names=in_names,
        out_names=out_names,
        normalization=NetworkAndLossNormalizationConfig(
            network=NormalizationConfig(
                means={n: 0.0 for n in all_names},
                stds={n: 1.0 for n in all_names},
            )
        ),
        saturation_normalization=saturation,
        global_mean_removal=global_mean_removal,
        residual_prediction=residual_prediction,
    )


def test_validation_zero_match_raises():
    names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    with pytest.raises(ValueError, match="matched none"):
        _step_config(
            [SaturationNormalizationConfig(names=["does_not_exist_*"])], names, names
        )


def test_validation_prediction_requires_output():
    in_names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    out_names = ["air_temperature_0"]
    with pytest.raises(ValueError, match="not an output"):
        _step_config(
            [
                SaturationNormalizationConfig(
                    names=["specific_total_water_0"], prediction=True
                )
            ],
            in_names,
            out_names,
        )


def test_validation_input_requires_input():
    in_names = ["air_temperature_0", "PRESsfc"]
    out_names = ["specific_total_water_0", "air_temperature_0"]
    with pytest.raises(ValueError, match="not an input"):
        _step_config(
            [
                SaturationNormalizationConfig(
                    names=["specific_total_water_0"], input="replace"
                )
            ],
            in_names,
            out_names,
        )


def test_validation_duplicate_field_across_entries_raises():
    names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    with pytest.raises(ValueError, match="more than one entry"):
        _step_config(
            [
                SaturationNormalizationConfig(
                    names=["specific_total_water_0"], input="append"
                ),
                SaturationNormalizationConfig(
                    names=["specific_total_water_*"], input="append"
                ),
            ],
            names,
            names,
        )


def test_validation_overlap_with_qsat_scaled_names_raises():
    names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    gmr = SharedGlobalMeanRemovalConfig(
        reference_field="air_temperature_0",
        field_names=[],
        qsat_scaled_names=["specific_total_water_0"],
    )
    with pytest.raises(ValueError, match="must not overlap"):
        _step_config(
            [
                SaturationNormalizationConfig(
                    names=["specific_total_water_0"], input="append"
                )
            ],
            names,
            names,
            global_mean_removal=gmr,
        )


def test_validation_derived_name_collision_raises():
    # a real relative_humidity_0 field collides with the derived channel
    names = [
        "specific_total_water_0",
        "relative_humidity_0",
        "air_temperature_0",
        "PRESsfc",
    ]
    with pytest.raises(ValueError, match="collides with an existing field"):
        _step_config(
            [
                SaturationNormalizationConfig(
                    names=["specific_total_water_0"], input="append"
                )
            ],
            names,
            names,
        )


def test_validation_residual_requires_coherent_prognostic():
    names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    # predict RH but feed q: ill-defined residual base
    with pytest.raises(ValueError, match="residual_prediction"):
        _step_config(
            [
                SaturationNormalizationConfig(
                    names=["specific_total_water_0"], prediction=True, input="none"
                )
            ],
            names,
            names,
            residual_prediction=True,
        )


def test_validation_prediction_append_allowed_without_residual():
    # without residual prediction, prediction and input are fully independent
    names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    _step_config(
        [
            SaturationNormalizationConfig(
                names=["specific_total_water_0"], prediction=True, input="append"
            )
        ],
        names,
        names,
        residual_prediction=False,
    )


# ── end-to-end through a stepper ─────────────────────────────────────────

_IMG_SHAPE = (16, 32)
_SFNO = {"scale_factor": 1, "embed_dim": 4, "num_layers": 2}


def _realistic_input(names):
    data = {}
    for name in names:
        if name.startswith("air_temperature") or name in (
            "surface_temperature",
            "TMP2m",
        ):
            data[name] = torch.full((1, *_IMG_SHAPE), 285.0)
        elif name in ("PRESsfc", "PS"):
            data[name] = torch.full((1, *_IMG_SHAPE), 1.0e5)
        elif name.startswith("specific_total_water"):
            data[name] = torch.full((1, *_IMG_SHAPE), 0.005)
        else:
            data[name] = torch.rand(1, *_IMG_SHAPE)
    return move_tensordict_to_device(data)


def _stepper(saturation, in_names, out_names, humidity_mean=0.0):
    all_names = sorted(set(in_names) | set(out_names))
    dataset_info = get_dataset_info(
        img_shape=_IMG_SHAPE, vertical_coordinate=_vertical_coordinate()
    )
    config = SingleModuleStepConfig(
        builder=ModuleSelector(type="SphericalFourierNeuralOperatorNet", config=_SFNO),
        in_names=in_names,
        out_names=out_names,
        normalization=NetworkAndLossNormalizationConfig(
            network=NormalizationConfig(
                means={
                    n: (humidity_mean if n.startswith("specific_total_water") else 0.0)
                    for n in all_names
                },
                stds={
                    n: (0.002 if n.startswith("specific_total_water") else 1.0)
                    for n in all_names
                },
            )
        ),
        saturation_normalization=saturation,
    )
    return config.get_step(dataset_info, lambda _: None)


def test_append_mode_runs_and_adds_derived_input_channel():
    in_names = ["specific_total_water_1", "air_temperature_1", "PRESsfc"]
    step = _stepper(
        [
            SaturationNormalizationConfig(
                names=["specific_total_water_1"], input="append"
            )
        ],
        in_names,
        in_names,
    )
    # derived channel is an internal input; the raw q input remains
    assert _RH1 in step.in_packer.names
    assert "specific_total_water_1" in step.in_packer.names
    assert _RH1 not in step.out_packer.names
    data = _realistic_input(step.input_names)
    output, _ = step.step(
        args=StepArgs(input=data, next_step_input_data=data, labels=None)
    )
    for name in in_names:
        assert output[name].shape == (1, *_IMG_SHAPE)
        assert torch.isfinite(output[name]).all()
    # the derived channel is not exposed in the public output
    assert _RH1 not in output


def test_prediction_replace_runs_with_derived_channel_in_packers():
    in_names = ["specific_total_water_1", "air_temperature_1", "PRESsfc"]
    step = _stepper(
        [
            SaturationNormalizationConfig(
                names=["specific_total_water_1"],
                prediction=True,
                input="replace",
            )
        ],
        in_names,
        in_names,
        humidity_mean=0.005,
    )
    # source q is replaced by the derived RH channel on both packers
    assert _RH1 in step.in_packer.names
    assert _RH1 in step.out_packer.names
    assert "specific_total_water_1" not in step.in_packer.names
    assert "specific_total_water_1" not in step.out_packer.names
    # the derived channel has identity normalizer statistics
    torch.testing.assert_close(step.normalizer.means[_RH1].cpu(), torch.tensor(0.0))
    torch.testing.assert_close(step.normalizer.stds[_RH1].cpu(), torch.tensor(1.0))
    data = _realistic_input(step.input_names)
    output, _ = step.step(
        args=StepArgs(input=data, next_step_input_data=data, labels=None)
    )
    # public output is physical q, not the derived channel
    assert output["specific_total_water_1"].shape == (1, *_IMG_SHAPE)
    assert torch.isfinite(output["specific_total_water_1"]).all()
    assert _RH1 not in output


def test_prediction_with_q_input_runs_without_residual():
    # prediction=True, input=none is allowed when not using residual prediction
    in_names = ["specific_total_water_1", "air_temperature_1", "PRESsfc"]
    step = _stepper(
        [
            SaturationNormalizationConfig(
                names=["specific_total_water_1"], prediction=True, input="none"
            )
        ],
        in_names,
        in_names,
    )
    assert "specific_total_water_1" in step.in_packer.names  # q input kept
    assert _RH1 in step.out_packer.names  # predicted as RH
    data = _realistic_input(step.input_names)
    output, _ = step.step(
        args=StepArgs(input=data, next_step_input_data=data, labels=None)
    )
    assert torch.isfinite(output["specific_total_water_1"]).all()
    assert _RH1 not in output
