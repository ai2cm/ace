import dataclasses

import dacite
import pytest
import torch

from fme.core.device import move_tensordict_to_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.normalizer import NormalizationConfig
from fme.core.step.global_mean_relaxation import (
    GlobalMeanRelaxationConfig,
    GlobalMeanRelaxationVariableConfig,
)


def _make_relaxation(
    variables: dict[str, GlobalMeanRelaxationVariableConfig],
    *,
    means: dict[str, float] | None = None,
    stds: dict[str, float] | None = None,
    area_weights: torch.Tensor | None = None,
):
    if means is None:
        means = {name: 0.0 for name in variables}
    if stds is None:
        stds = {name: 1.0 for name in variables}
    if area_weights is None:
        area_weights = torch.ones(4, 5)
    normalizer = NormalizationConfig(means=means, stds=stds).build(list(means))
    ops = LatLonOperations(area_weights=area_weights)
    return GlobalMeanRelaxationConfig(variables=variables).build(
        gridded_operations=ops, normalizer=normalizer
    )


def test_timescale_steps_must_be_positive():
    with pytest.raises(ValueError, match="timescale_steps must be positive"):
        GlobalMeanRelaxationVariableConfig(target=1.0, timescale_steps=0.0)
    with pytest.raises(ValueError, match="timescale_steps must be positive"):
        GlobalMeanRelaxationVariableConfig(target=1.0, timescale_steps=-1.0)


def test_target_string_must_be_mean():
    # The arg-type suppression below is intentional: this test exercises
    # runtime validation against an invalid Literal value, which mypy
    # correctly rejects at static-typing time.
    with pytest.raises(ValueError, match="target must be a number or 'mean'"):
        GlobalMeanRelaxationVariableConfig(target="median", timescale_steps=10.0)  # type: ignore[arg-type]


def test_empty_variables_raises():
    with pytest.raises(ValueError, match="at least one entry"):
        GlobalMeanRelaxationConfig(variables={})


def test_validate_names_missing_raises():
    config = GlobalMeanRelaxationConfig(
        variables={
            "a": GlobalMeanRelaxationVariableConfig(target=0.0, timescale_steps=2.0)
        }
    )
    with pytest.raises(ValueError, match="missing.*'a'"):
        config.validate_names(out_names=["b"])


def test_mean_target_requires_normalization_mean():
    normalizer = NormalizationConfig(means={"other": 0.0}, stds={"other": 1.0}).build(
        ["other"]
    )
    ops = LatLonOperations(area_weights=torch.ones(4, 5))
    config = GlobalMeanRelaxationConfig(
        variables={
            "a": GlobalMeanRelaxationVariableConfig(target="mean", timescale_steps=2.0)
        }
    )
    with pytest.raises(ValueError, match="requires 'a' to have a normalization mean"):
        config.build(gridded_operations=ops, normalizer=normalizer)


def test_worked_example_constant_field():
    # mean=10, target=4, tau=2 -> offset = (10-4)/2 = 3, result = 10-3 = 7
    relaxation = _make_relaxation(
        variables={
            "a": GlobalMeanRelaxationVariableConfig(target=4.0, timescale_steps=2.0)
        }
    )
    data = move_tensordict_to_device({"a": torch.full((1, 4, 5), 10.0)})
    out = relaxation(data)
    torch.testing.assert_close(out["a"], torch.full_like(out["a"], 7.0))


def test_uses_normalization_mean_when_target_is_mean():
    relaxation = _make_relaxation(
        variables={
            "a": GlobalMeanRelaxationVariableConfig(target="mean", timescale_steps=2.0)
        },
        means={"a": 4.0},
        stds={"a": 1.0},
    )
    data = move_tensordict_to_device({"a": torch.full((1, 4, 5), 10.0)})
    out = relaxation(data)
    # mean=10, target=4 (normalization mean), tau=2 -> result = 7
    torch.testing.assert_close(out["a"], torch.full_like(out["a"], 7.0))


def test_preserves_spatial_pattern():
    # Relaxation only shifts the field uniformly, so all gradients survive.
    relaxation = _make_relaxation(
        variables={
            "a": GlobalMeanRelaxationVariableConfig(target=0.0, timescale_steps=5.0)
        }
    )
    field = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5)
    data = move_tensordict_to_device({"a": field.clone()})
    out = relaxation(data)
    diff = out["a"] - data["a"]
    # The offset is constant across spatial dims for each sample.
    torch.testing.assert_close(diff, torch.full_like(diff, diff.flatten()[0].item()))


def test_unconfigured_variables_passthrough():
    relaxation = _make_relaxation(
        variables={
            "a": GlobalMeanRelaxationVariableConfig(target=0.0, timescale_steps=2.0)
        },
        means={"a": 0.0, "b": 0.0},
        stds={"a": 1.0, "b": 1.0},
    )
    a = torch.full((1, 4, 5), 10.0)
    b = torch.full((1, 4, 5), 99.0)
    data = move_tensordict_to_device({"a": a, "b": b})
    out = relaxation(data)
    torch.testing.assert_close(out["b"], b.to(out["b"].device))


def test_per_sample_relaxation():
    relaxation = _make_relaxation(
        variables={
            "a": GlobalMeanRelaxationVariableConfig(target=0.0, timescale_steps=4.0)
        }
    )
    field = torch.stack(
        [torch.full((4, 5), 8.0), torch.full((4, 5), 0.0)]
    )  # shape [2, 4, 5]
    data = move_tensordict_to_device({"a": field})
    out = relaxation(data)
    # sample 0: mean=8, offset=2 -> 6 everywhere; sample 1: mean=0 -> 0 everywhere
    torch.testing.assert_close(out["a"][0], torch.full_like(out["a"][0], 6.0))
    torch.testing.assert_close(out["a"][1], torch.full_like(out["a"][1], 0.0))


def test_config_roundtrip_via_dacite():
    config = GlobalMeanRelaxationConfig(
        variables={
            "a": GlobalMeanRelaxationVariableConfig(target=1.5, timescale_steps=10.0),
            "b": GlobalMeanRelaxationVariableConfig(target="mean", timescale_steps=5.0),
        }
    )
    data = dataclasses.asdict(config)
    restored = dacite.from_dict(
        data_class=GlobalMeanRelaxationConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
    assert restored == config
