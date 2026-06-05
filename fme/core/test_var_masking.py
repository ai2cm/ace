import pytest
import torch

from fme.core.var_masking import UniformMaskingConfig, VariableMaskingConfig


@pytest.mark.parametrize(
    "per_variable,match",
    [
        ({"default_rate": -0.1}, "default_rate"),
        ({"default_rate": 1.1}, "default_rate"),
        ({"T": -0.1}, "'T'"),
        ({"T": 1.5}, "'T'"),
    ],
)
def test_variable_masking_config_invalid_rate_raises(per_variable, match):
    with pytest.raises(ValueError, match=match):
        VariableMaskingConfig(per_variable=per_variable)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"min_vars": -1}, "min_vars"),
        ({"max_vars": -1}, "max_vars"),
        ({"min_vars": 3, "max_vars": 1}, "min_vars"),
    ],
)
def test_uniform_config_invalid_bounds_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        UniformMaskingConfig(**kwargs)


def test_uniform_config_bounds_validated_against_available_variables():
    config = VariableMaskingConfig(uniform=UniformMaskingConfig(min_vars=5))
    with pytest.raises(ValueError, match="min_vars"):
        config.sample_masks(["a", "b"], batch_size=4, n_ensemble=1)


def test_sample_masks_rate_zero_returns_empty():
    config = VariableMaskingConfig(per_variable={"default_rate": 0.0, "T": 0.0})
    masks = config.sample_masks(["T", "q"], batch_size=8, n_ensemble=1)
    assert masks == {}


def test_sample_masks_exact_rate_one_always_masks():
    config = VariableMaskingConfig(per_variable={"T": 1.0})
    masks = config.sample_masks(["T", "q"], batch_size=4, n_ensemble=1)
    assert "T" in masks
    assert not masks["T"].any()
    assert "q" not in masks


def test_sample_masks_prefix_groups_levels_together():
    torch.manual_seed(0)
    config = VariableMaskingConfig(per_variable={"air_temperature_": 0.5})
    names = ["air_temperature_0", "air_temperature_1", "air_temperature_2", "q"]
    masks = config.sample_masks(names, batch_size=64, n_ensemble=1)
    assert set(masks.keys()) == set(names[:-1])
    assert (masks["air_temperature_0"] == masks["air_temperature_1"]).all()
    assert (masks["air_temperature_0"] == masks["air_temperature_2"]).all()
    assert not torch.equal(masks["air_temperature_0"], torch.ones(64, dtype=torch.bool))


def test_sample_masks_default_rate_masks_unmatched_variables_independently():
    torch.manual_seed(0)
    config = VariableMaskingConfig(per_variable={"default_rate": 0.5})
    masks = config.sample_masks(["T", "q"], batch_size=64, n_ensemble=1)
    assert set(masks) == {"T", "q"}
    assert not torch.equal(masks["T"], masks["q"])


def test_sample_masks_zero_rate_variable_never_masked():
    config = VariableMaskingConfig(per_variable={"T": 1.0, "SST": 0.0})
    masks = config.sample_masks(["T", "SST", "q"], batch_size=4, n_ensemble=1)
    assert "T" in masks
    assert "SST" not in masks


def test_uniform_min_max_strings_accepted():
    config = UniformMaskingConfig(min_vars="min", max_vars="max")
    assert config.min_vars == "min"
    assert config.max_vars == "max"


def test_uniform_only_ignore_vars_excluded():
    config = VariableMaskingConfig(
        uniform=UniformMaskingConfig(min_vars=1, max_vars=2, ignore_vars=["SST"])
    )
    masks = config.sample_masks(["T", "q", "SST"], batch_size=32, n_ensemble=1)
    assert "SST" not in masks


def test_uniform_only_mask_count_in_range():
    config = VariableMaskingConfig(uniform=UniformMaskingConfig(min_vars=1, max_vars=2))
    names = ["a", "b", "c", "d"]
    masks = config.sample_masks(names, batch_size=100, n_ensemble=1)
    for i in range(100):
        n_masked = sum(
            1 for name in names if name in masks and not masks[name][i].item()
        )
        assert 1 <= n_masked <= 2


def test_uniform_only_min_max_zero_returns_empty():
    config = VariableMaskingConfig(uniform=UniformMaskingConfig(min_vars=0, max_vars=0))
    masks = config.sample_masks(["a", "b"], batch_size=4, n_ensemble=1)
    assert masks == {}


def test_uniform_only_max_vars_max_string():
    names = ["a", "b", "c"]
    config = VariableMaskingConfig(
        uniform=UniformMaskingConfig(min_vars=len(names), max_vars="max")
    )
    masks = config.sample_masks(names, batch_size=8, n_ensemble=1)
    for name in names:
        assert name in masks
        assert not masks[name].any()


def test_combined_exact_rate_excluded_from_uniform_pool():
    config = VariableMaskingConfig(
        uniform=UniformMaskingConfig(min_vars=1, max_vars=1),
        per_variable={"global_mean_co2": 1.0},
    )
    names = ["global_mean_co2", "a", "b"]
    masks = config.sample_masks(names, batch_size=32, n_ensemble=1)
    assert not masks["global_mean_co2"].any()
    for i in range(32):
        n_uniform_masked = sum(
            1 for name in ["a", "b"] if name in masks and not masks[name][i].item()
        )
        assert n_uniform_masked == 1


def test_combined_zero_rate_exact_excluded_from_masks_and_uniform_pool():
    config = VariableMaskingConfig(
        uniform=UniformMaskingConfig(min_vars=2, max_vars=2),
        per_variable={"global_mean_co2": 0.0},
    )
    names = ["global_mean_co2", "a", "b"]
    masks = config.sample_masks(names, batch_size=8, n_ensemble=1)
    assert "global_mean_co2" not in masks
    assert not masks["a"].any()
    assert not masks["b"].any()


def test_uniform_ignore_vars_do_not_affect_explicit_per_variable_rates():
    config = VariableMaskingConfig(
        uniform=UniformMaskingConfig(
            min_vars=1, max_vars=1, ignore_vars=["SST", "ignored"]
        ),
        per_variable={"SST": 1.0},
    )
    masks = config.sample_masks(
        ["SST", "ignored", "eligible"], batch_size=8, n_ensemble=1
    )
    assert not masks["SST"].any()
    assert "ignored" not in masks
    assert not masks["eligible"].any()


def test_default_rate_excludes_unmatched_variables_from_uniform_pool():
    config = VariableMaskingConfig(
        uniform=UniformMaskingConfig(min_vars=1, max_vars=1),
        per_variable={"default_rate": 1.0},
    )
    with pytest.raises(ValueError, match="eligible variables"):
        config.sample_masks(["a", "b"], batch_size=4, n_ensemble=1)


def test_zero_default_rate_leaves_unmatched_variables_in_uniform_pool():
    config = VariableMaskingConfig(
        uniform=UniformMaskingConfig(min_vars=2, max_vars=2),
        per_variable={"default_rate": 0.0},
    )
    masks = config.sample_masks(["a", "b"], batch_size=4, n_ensemble=1)
    assert not masks["a"].any()
    assert not masks["b"].any()


def test_sample_masks_device_placement():
    config = VariableMaskingConfig(per_variable={"T": 1.0})
    masks = config.sample_masks(
        ["T"], batch_size=4, n_ensemble=1, device=torch.device("cpu")
    )
    assert masks["T"].device == torch.device("cpu")


def test_sample_masks_repeats_explicit_masks_for_ensemble_members():
    torch.manual_seed(2)
    n_samples = 6
    n_ensemble = 3
    config = VariableMaskingConfig(per_variable={"T": 0.5})
    masks = config.sample_masks(
        ["T"], batch_size=n_samples * n_ensemble, n_ensemble=n_ensemble
    )
    grouped = masks["T"].view(n_samples, n_ensemble)
    assert (grouped == grouped[:, :1]).all()
    assert not torch.equal(grouped[:, 0], torch.ones(n_samples, dtype=torch.bool))
    assert not torch.equal(grouped[:, 0], torch.zeros(n_samples, dtype=torch.bool))


def test_sample_masks_repeats_uniform_masks_for_ensemble_members():
    torch.manual_seed(1)
    n_samples = 8
    n_ensemble = 3
    names = ["a", "b", "c"]
    config = VariableMaskingConfig(uniform=UniformMaskingConfig(min_vars=1, max_vars=1))
    masks = config.sample_masks(
        names, batch_size=n_samples * n_ensemble, n_ensemble=n_ensemble
    )
    for name in names:
        grouped = masks[name].view(n_samples, n_ensemble)
        assert (grouped == grouped[:, :1]).all()
    for i in range(n_samples * n_ensemble):
        n_masked = sum(
            1 for name in names if name in masks and not masks[name][i].item()
        )
        assert n_masked == 1


def test_sample_masks_raises_if_batch_size_not_divisible_by_n_ensemble():
    config = VariableMaskingConfig(per_variable={"T": 0.5})
    with pytest.raises(ValueError, match="divisible"):
        config.sample_masks(["T"], batch_size=5, n_ensemble=3)
