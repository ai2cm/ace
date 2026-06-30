import pytest
import torch

from fme.core.var_masking import VariableMaskingConfig

DEVICE = torch.device("cpu")


def _names(n: int) -> list[str]:
    return [f"var_{i}" for i in range(n)]


def test_max_masked_vars_validation():
    with pytest.raises(ValueError, match="max_masked_vars"):
        VariableMaskingConfig(max_masked_vars=-1)


def test_variable_masking_rates_validation():
    with pytest.raises(ValueError, match="must be in"):
        VariableMaskingConfig(variable_masking_rates={"a": -0.1})
    with pytest.raises(ValueError, match="must be in"):
        VariableMaskingConfig(variable_masking_rates={"a": 1.1})


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


def test_bernoulli_rate_zero_keeps_all():
    config = VariableMaskingConfig(
        max_masked_vars=0, variable_masking_rates={"var_0": 0.0, "var_1": 0.0}
    )
    for _ in range(16):
        assert config.sample_mask(_names(4), DEVICE).all()


def test_rate_one_var_not_dropped_when_no_slots():
    """With max_masked_vars=0 there are no slots, so a rate-1 var is not dropped."""
    config = VariableMaskingConfig(
        max_masked_vars=0, variable_masking_rates={"var_0": 1.0}
    )
    names = _names(4)
    for _ in range(32):
        mask = config.sample_mask(names, DEVICE)
        assert mask.all(), "no uniform slots means nothing can be dropped"


def test_rate_one_var_always_dropped_with_slots():
    """With at least one slot, a rate-1 var is always dropped, count stays k."""
    config = VariableMaskingConfig(
        max_masked_vars=2, variable_masking_rates={"var_0": 1.0}
    )
    names = _names(5)
    for _ in range(128):
        mask = config.sample_mask(names, DEVICE)
        n_masked = int((~mask).sum().item())
        # The fired bernoulli var must be dropped whenever k >= 1; k may be 0.
        if n_masked == 0:
            continue
        assert not bool(mask[0, 0].item()), "rate-1 var must be dropped when k>=1"
        assert 1 <= n_masked <= 2


def test_fired_var_already_in_uniform_set_leaves_count_unchanged():
    """A fired bernoulli var that the uniform draw already dropped is a no-op.

    With max_masked_vars == n_channels and a single rate-1 var, the dropped
    count must always equal the uniform count k (never exceed it).
    """
    n = 4
    config = VariableMaskingConfig(
        max_masked_vars=n, variable_masking_rates={"var_0": 1.0}
    )
    names = _names(n)
    for _ in range(256):
        mask = config.sample_mask(names, DEVICE)
        n_masked = int((~mask).sum().item())
        assert 0 <= n_masked <= n
        if n_masked >= 1:
            assert not bool(mask[0, 0].item())


def test_multiple_fired_vars_capped_by_slots():
    """More fired vars than evictable slots: only k-worth dropped."""
    n = 5
    config = VariableMaskingConfig(
        max_masked_vars=1,
        variable_masking_rates={name: 1.0 for name in _names(n)},
    )
    names = _names(n)
    for _ in range(128):
        mask = config.sample_mask(names, DEVICE)
        # k is drawn from [0, 1]; all vars fire but only k can be dropped.
        assert int((~mask).sum().item()) <= 1
