import pytest
import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.device import move_tensordict_to_device
from fme.core.humidity import saturation_specific_humidity
from fme.core.step.saturation_normalization import (
    SaturationNormalizationConfig,
    _relative_humidity_name,
    build_saturation_normalization,
    resolve_all_fields,
)

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
#
# resolve_all_fields carries the per-rule validation (patterns, sides, level
# suffix, derived-name collisions, cross-rule duplicates). The step configs
# call it from their __post_init__; the step-level residual-coherence check and
# end-to-end routing are exercised in fme/ace/step/test_two_track.py.


def test_validation_zero_match_raises():
    names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    with pytest.raises(ValueError, match="matched none"):
        resolve_all_fields(
            [SaturationNormalizationConfig(names=["does_not_exist_*"])], names, names
        )


def test_validation_prediction_requires_output():
    in_names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    out_names = ["air_temperature_0"]
    with pytest.raises(ValueError, match="not an output"):
        resolve_all_fields(
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
        resolve_all_fields(
            [
                SaturationNormalizationConfig(
                    names=["specific_total_water_0"], input="replace"
                )
            ],
            in_names,
            out_names,
        )


def test_validation_field_without_level_suffix_raises():
    names = ["specific_total_water", "air_temperature_0", "PRESsfc"]
    with pytest.raises(ValueError, match="must end in '_<level>'"):
        resolve_all_fields(
            [SaturationNormalizationConfig(names=["specific_total_water"])],
            names,
            names,
        )


def test_validation_duplicate_field_across_entries_raises():
    names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    with pytest.raises(ValueError, match="more than one entry"):
        resolve_all_fields(
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


def test_validation_derived_name_collision_raises():
    # a real relative_humidity_0 field collides with the derived channel
    names = [
        "specific_total_water_0",
        "relative_humidity_0",
        "air_temperature_0",
        "PRESsfc",
    ]
    with pytest.raises(ValueError, match="collides with an existing field"):
        resolve_all_fields(
            [
                SaturationNormalizationConfig(
                    names=["specific_total_water_0"], input="append"
                )
            ],
            names,
            names,
        )


def test_validation_prediction_append_allowed():
    # prediction and input are independent knobs: predicting RH while feeding an
    # appended RH channel (raw q kept) resolves without error.
    names = ["specific_total_water_0", "air_temperature_0", "PRESsfc"]
    fields = resolve_all_fields(
        [
            SaturationNormalizationConfig(
                names=["specific_total_water_0"], prediction=True, input="append"
            )
        ],
        names,
        names,
    )
    assert len(fields) == 1
    assert fields[0].rh_out is True
    assert fields[0].rh_extra is True
