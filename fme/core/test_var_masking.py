import pytest
import torch

from fme.core.var_masking import MaskingGroupConfig, VariableMaskingConfig

DEVICE = torch.device("cpu")


def _names(n: int) -> list[str]:
    return [f"var_{i}" for i in range(n)]


def _group(variables: list[str], rate: float) -> MaskingGroupConfig:
    return MaskingGroupConfig(variables=variables, rate=rate)


def test_max_masked_vars_validation():
    with pytest.raises(ValueError, match="max_masked_vars"):
        VariableMaskingConfig(max_masked_vars=-1)
    with pytest.raises(ValueError, match="max_masked_vars"):
        VariableMaskingConfig(max_masked_vars=True)


def test_group_rate_validation():
    with pytest.raises(ValueError, match="rate must be in"):
        VariableMaskingConfig(variable_masking_groups=[_group(["a"], -0.1)])
    with pytest.raises(ValueError, match="rate must be in"):
        VariableMaskingConfig(variable_masking_groups=[_group(["a"], 1.1)])


def test_empty_group_validation():
    with pytest.raises(ValueError, match="non-empty"):
        VariableMaskingConfig(variable_masking_groups=[_group([], 0.5)])


def test_duplicate_variable_across_groups_validation():
    with pytest.raises(ValueError, match="more than one masking group"):
        VariableMaskingConfig(
            variable_masking_groups=[_group(["a", "b"], 0.5), _group(["b"], 0.5)]
        )


def test_validate_names_rejects_unknown_group_variable():
    config = VariableMaskingConfig(
        variable_masking_groups=[_group(["var_0", "typo"], 0.5)]
    )
    with pytest.raises(ValueError, match="not in packed input channels"):
        config.validate_names(_names(3))


def test_validate_names_accepts_known_group_variables():
    config = VariableMaskingConfig(
        variable_masking_groups=[_group(["var_0", "var_2"], 0.5)]
    )
    config.validate_names(_names(3))


def test_validate_names_noop_without_groups():
    VariableMaskingConfig(max_masked_vars=2).validate_names(_names(3))


def test_sample_mask_shape_and_dtype():
    config = VariableMaskingConfig(max_masked_vars=3)
    mask = config.sample_mask(_names(10), DEVICE)
    assert mask.shape == (1, 10)
    assert mask.dtype == torch.bool


def test_uniform_count_in_range():
    config = VariableMaskingConfig(max_masked_vars=4)
    names = _names(8)
    counts = set()
    for _ in range(256):
        mask = config.sample_mask(names, DEVICE)
        n_masked = int((~mask).sum().item())
        assert 0 <= n_masked <= 4
        counts.add(n_masked)
    # min count is hard-coded 0, so zero-drop draws must be possible
    assert 0 in counts


def test_uniform_count_capped_at_n_channels():
    config = VariableMaskingConfig(max_masked_vars=100)
    names = _names(5)
    for _ in range(64):
        mask = config.sample_mask(names, DEVICE)
        assert 0 <= int((~mask).sum().item()) <= 5


def test_no_masking_by_default():
    config = VariableMaskingConfig()
    for _ in range(16):
        assert config.sample_mask(_names(5), DEVICE).all()


def test_group_rate_zero_keeps_all():
    config = VariableMaskingConfig(
        max_masked_vars=0, variable_masking_groups=[_group(["var_0", "var_1"], 0.0)]
    )
    for _ in range(16):
        assert config.sample_mask(_names(4), DEVICE).all()


def test_rate_one_group_dropped_with_no_uniform_slots():
    """A rate-1 group is dropped solely by its Bernoulli, even with no uniform."""
    config = VariableMaskingConfig(
        max_masked_vars=0, variable_masking_groups=[_group(["var_0"], 1.0)]
    )
    names = _names(4)
    for _ in range(32):
        mask = config.sample_mask(names, DEVICE)
        assert not bool(mask[0, 0].item()), "rate-1 group must always be dropped"
        # only var_0 fires and max_masked_vars=0, so nothing else is dropped
        assert int((~mask).sum().item()) == 1


def test_rate_one_group_always_dropped():
    """A rate-1 group is always dropped regardless of the uniform count k."""
    config = VariableMaskingConfig(
        max_masked_vars=2, variable_masking_groups=[_group(["var_0"], 1.0)]
    )
    names = _names(5)
    for _ in range(128):
        mask = config.sample_mask(names, DEVICE)
        assert not bool(mask[0, 0].item()), "rate-1 group must always be dropped"
        # var_0 excluded from uniform pool of 4; dropped count is 1 plus k in [0, 2].
        assert 1 <= int((~mask).sum().item()) <= 3


def test_grouped_vars_excluded_from_uniform_pool():
    """Grouped channels are never dropped by the uniform mechanism.

    A rate-0 grouped var must stay present no matter how large max_masked_vars
    is, because it is excluded from the uniform pool entirely.
    """
    n = 4
    config = VariableMaskingConfig(
        max_masked_vars=n, variable_masking_groups=[_group(["var_0"], 0.0)]
    )
    names = _names(n)
    for _ in range(256):
        mask = config.sample_mask(names, DEVICE)
        assert bool(mask[0, 0].item()), "rate-0 grouped var must never be dropped"
        # uniform pool is var_1..var_3 (3 channels), so at most 3 are dropped
        assert 0 <= int((~mask).sum().item()) <= n - 1


def test_uniform_count_capped_at_pool_size():
    """The uniform count is capped at the number of ungrouped channels."""
    n = 5
    config = VariableMaskingConfig(
        max_masked_vars=100,
        variable_masking_groups=[_group(["var_0", "var_1"], 0.0)],
    )
    names = _names(n)
    for _ in range(64):
        mask = config.sample_mask(names, DEVICE)
        # var_0, var_1 are rate-0 and excluded; pool is the other 3 channels
        assert bool(mask[0, 0].item()) and bool(mask[0, 1].item())
        assert 0 <= int((~mask).sum().item()) <= 3


def test_marginal_rate_independent_of_uniform():
    """A group's drop frequency matches its rate, independent of uniform."""
    n = 6
    rate = 0.5
    config = VariableMaskingConfig(
        max_masked_vars=n, variable_masking_groups=[_group(["var_0"], rate)]
    )
    names = _names(n)
    torch.manual_seed(0)
    trials = 4000
    dropped = sum(
        int(not bool(config.sample_mask(names, DEVICE)[0, 0].item()))
        for _ in range(trials)
    )
    freq = dropped / trials
    assert abs(freq - rate) < 0.05, f"expected ~{rate}, got {freq}"


def test_group_members_dropped_together():
    """A fired multi-var group drops all its members jointly, never split."""
    config = VariableMaskingConfig(
        max_masked_vars=0,
        variable_masking_groups=[_group(["var_0", "var_1", "var_2"], 0.5)],
    )
    names = _names(4)
    torch.manual_seed(0)
    saw_dropped = False
    saw_present = False
    for _ in range(256):
        mask = config.sample_mask(names, DEVICE)
        members = [bool(mask[0, i].item()) for i in range(3)]
        # all three members share one draw: all present or all absent
        assert all(members) or not any(members), members
        # var_3 is ungrouped and max_masked_vars=0, so it is never dropped
        assert bool(mask[0, 3].item())
        saw_dropped |= not any(members)
        saw_present |= all(members)
    assert saw_dropped and saw_present


def test_multiple_groups_fire_independently():
    """Distinct groups drop on their own draws; a fired group drops all members."""
    n = 5
    config = VariableMaskingConfig(
        max_masked_vars=0,
        variable_masking_groups=[
            _group(["var_0", "var_1"], 1.0),
            _group(["var_2", "var_3"], 1.0),
        ],
    )
    names = _names(n)
    for _ in range(128):
        mask = config.sample_mask(names, DEVICE)
        # both rate-1 groups fire; all four grouped vars dropped, var_4 kept
        assert int((~mask).sum().item()) == 4
        assert bool(mask[0, 4].item())
