import random

import numpy as np
import pytest
import torch

from fme.ace.data_loading.augmentation import (
    AugmentationConfig,
    RotateModifier,
    VariableMaskingConfig,
    VariableMaskingModifier,
)
from fme.ace.data_loading.batch_data import BatchData


def rotate(data: torch.Tensor) -> torch.Tensor:
    return torch.flip(data, dims=[-2, -1])


def test_rotate_modifier_all_rotation():
    rotate_modifier = RotateModifier(
        rotate_probability=1.0, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    data_mask = {"PS": torch.tensor([True])}
    batch = BatchData.new_for_testing(
        names=["UGRD", "VGRD", "PS"],
        n_samples=1,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
        data_mask=data_mask,
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data["UGRD"].shape == (1, 2, n_lat, n_lon)
    assert torch.allclose(rotate(rotated_batch.data["UGRD"]), -1 * batch.data["UGRD"])
    assert torch.allclose(rotate(rotated_batch.data["VGRD"]), -1 * batch.data["VGRD"])
    assert torch.allclose(rotate(rotated_batch.data["PS"]), batch.data["PS"])
    assert rotated_batch.data_mask is not None
    torch.testing.assert_close(rotated_batch.data_mask["PS"], data_mask["PS"])


def test_rotate_modifier_no_rotation():
    rotate_modifier = RotateModifier(
        rotate_probability=0.0, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData.new_for_testing(
        names=["UGRD", "VGRD", "PS"],
        n_samples=1,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data["UGRD"].shape == (1, 2, n_lat, n_lon)
    assert torch.allclose(rotated_batch.data["UGRD"], batch.data["UGRD"])
    assert torch.allclose(rotated_batch.data["VGRD"], batch.data["VGRD"])
    assert torch.allclose(rotated_batch.data["PS"], batch.data["PS"])


def test_rotate_modifier_random_rotation():
    random.seed(0)
    rotate_modifier = RotateModifier(
        rotate_probability=0.5, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData.new_for_testing(
        names=["UGRD", "VGRD", "PS"],
        n_samples=40,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data.keys() == batch.data.keys()
    assert rotated_batch.data["UGRD"].shape == (40, 2, n_lat, n_lon)
    rotated = {}
    unrotated = {}
    for name in rotated_batch.data:
        unrotated[name] = np.all(
            torch.abs(batch.data[name] - rotated_batch.data[name]).cpu().numpy() < 1e-6,
            axis=(1, 2, 3),
        )
        if name in ("UGRD", "VGRD"):
            sign = -1
        else:
            sign = 1
        rotated[name] = np.all(
            torch.abs(sign * rotate(batch.data[name]) - rotated_batch.data[name])
            .cpu()
            .numpy()
            < 1e-6,
            axis=(1, 2, 3),
        )
        assert np.all(rotated[name] + unrotated[name] == 1), name
        assert np.sum(rotated[name]) > 0, name
        assert np.sum(unrotated[name]) > 0, name
    for name in ("VGRD", "PS"):
        assert np.all(rotated[name] == rotated["UGRD"]), name
        assert np.all(unrotated[name] == unrotated["UGRD"]), name


@pytest.mark.parametrize(
    "name, additional_directional_names, match_expected",
    [
        ("UGRD", [], True),
        ("VGRD", [], True),
        ("UGRD_10m", [], True),
        ("UGRD_10m", ["UGRD"], True),
        ("VGRD200", [], True),
        ("eastward_wind_3", [], True),
        ("UGRD10m", [], True),
        ("NWIND10m", [], False),
        ("NWIND10m", ["NWIND"], True),
    ],
)
def test_rotate_modifier_pattern(
    name: str, additional_directional_names: list[str], match_expected: bool
):
    rotate_modifier = RotateModifier(
        rotate_probability=1.0,
        additional_directional_names=additional_directional_names,
    )
    assert (rotate_modifier._pattern.match(name) is not None) == match_expected, name


# ── VariableMaskingModifier tests ────────────────────────────────────────────


def _masking_batch(names, n_samples=8, n_timesteps=2):
    return BatchData.new_for_testing(
        names=names,
        n_samples=n_samples,
        n_timesteps=n_timesteps,
        device=torch.device("cpu"),
    )


def test_variable_masking_rate_zero_never_masks():
    modifier = VariableMaskingModifier(
        VariableMaskingConfig(default_rate=0.0, rates={"T": 0.0})
    )
    batch = _masking_batch(["T", "q"])
    result = modifier(batch)
    assert result.data_mask is None
    assert not result.data["T"].isnan().any()


def test_variable_masking_rate_one_always_masks():
    modifier = VariableMaskingModifier(VariableMaskingConfig(rates={"T": 1.0}))
    batch = _masking_batch(["T", "q"], n_timesteps=2)
    result = modifier(batch)
    # Only the IC (first) timestep is masked; the target timestep stays real.
    assert result.data["T"][:, 0].isnan().all()
    assert not result.data["T"][:, 1:].isnan().any()
    assert not result.data["q"].isnan().any()
    # data_mask is not modified by variable masking augmentation
    assert result.data_mask is None


def test_variable_masking_prefix_masks_all_levels_together():
    """Variables matched by the same prefix key share one random draw."""
    modifier = VariableMaskingModifier(
        VariableMaskingConfig(rates={"air_temperature_": 0.5})
    )
    names = ["air_temperature_0", "air_temperature_1", "air_temperature_2"]
    batch = _masking_batch(names, n_samples=40)
    result = modifier(batch)

    # Determine which samples were masked by checking NaN in IC (t=0)
    masked_0 = result.data["air_temperature_0"][:, 0].isnan().any(dim=(-2, -1))
    masked_1 = result.data["air_temperature_1"][:, 0].isnan().any(dim=(-2, -1))
    masked_2 = result.data["air_temperature_2"][:, 0].isnan().any(dim=(-2, -1))
    # All levels must be masked/unmasked together
    assert (masked_0 == masked_1).all()
    assert (masked_0 == masked_2).all()
    # With 40 samples and rate=0.5, some should be masked and some not
    assert masked_0.any()
    assert (~masked_0).any()
    # Target timesteps must never be NaN
    for name in names:
        assert not result.data[name][:, 1:].isnan().any()


def test_variable_masking_prefix_and_exact_are_independent():
    """A prefix-grouped variable and an exact-named variable are drawn independently."""
    modifier = VariableMaskingModifier(
        VariableMaskingConfig(rates={"air_temperature_": 1.0, "PRESsfc": 1.0})
    )
    names = ["air_temperature_0", "air_temperature_1", "PRESsfc"]
    batch = _masking_batch(names, n_samples=4)
    result = modifier(batch)

    # IC timestep should be fully masked; target timesteps intact
    assert result.data["air_temperature_0"][:, 0].isnan().all()
    assert result.data["air_temperature_1"][:, 0].isnan().all()
    assert result.data["PRESsfc"][:, 0].isnan().all()
    assert not result.data["air_temperature_0"][:, 1:].isnan().any()
    assert not result.data["air_temperature_1"][:, 1:].isnan().any()
    assert not result.data["PRESsfc"][:, 1:].isnan().any()


def test_variable_masking_default_rate_applies_to_unlisted():
    modifier = VariableMaskingModifier(
        VariableMaskingConfig(default_rate=1.0, rates={"T": 0.0})
    )
    batch = _masking_batch(["T", "q", "u"])
    result = modifier(batch)
    # T has rate 0 → never masked
    assert not result.data["T"].isnan().any()
    # q and u have default_rate=1.0 → IC timestep always masked, targets intact
    assert result.data["q"][:, 0].isnan().all()
    assert result.data["u"][:, 0].isnan().all()
    assert not result.data["q"][:, 1:].isnan().any()
    assert not result.data["u"][:, 1:].isnan().any()


def test_variable_masking_only_ic_timesteps_masked():
    """Only the first n_ic_timesteps are NaN; target timesteps are always real."""
    modifier = VariableMaskingModifier(
        VariableMaskingConfig(rates={"T": 0.5}), n_ic_timesteps=2
    )
    batch = _masking_batch(["T"], n_samples=20, n_timesteps=4)
    result = modifier(batch)
    # data_mask is not modified by augmentation
    assert result.data_mask is None
    for sample_idx in range(20):
        ic_nan = result.data["T"][sample_idx, :2].isnan().all().item()
        tgt_nan = result.data["T"][sample_idx, 2:].isnan().any().item()
        # Either all IC timesteps are NaN (masked) or none are; targets never NaN
        assert not tgt_nan, f"sample {sample_idx}: target timestep has NaN"
        if ic_nan:
            assert result.data["T"][sample_idx, :2].isnan().all()


def test_augmentation_config_builds_composed_modifier():
    """Both rotate and variable_masking configured → ComposedModifier applied."""
    from fme.ace.data_loading.augmentation import ComposedModifier

    config = AugmentationConfig(
        rotate_probability=1.0,
        variable_masking=VariableMaskingConfig(rates={"T": 1.0}),
    )
    modifier = config.build_modifier()
    assert isinstance(modifier, ComposedModifier)
    batch = _masking_batch(["T", "UGRD", "VGRD"], n_samples=2)
    result = modifier(batch)
    # Variable masking: only IC (first) timestep of T should be NaN
    assert result.data["T"][:, 0].isnan().all()
    assert not result.data["T"][:, 1:].isnan().any()
    # RotateModifier should have flipped spatial dims (all samples at rate=1)
    assert torch.allclose(
        torch.flip(result.data["UGRD"], dims=[-2, -1]), -1 * batch.data["UGRD"]
    )
