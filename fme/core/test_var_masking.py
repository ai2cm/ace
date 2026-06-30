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


def test_rate_one_var_dropped_with_no_uniform_slots():
    """With max_masked_vars=0 a rate-1 var is still dropped via the OR."""
    config = VariableMaskingConfig(
        max_masked_vars=0, variable_masking_rates={"var_0": 1.0}
    )
    names = _names(4)
    for _ in range(32):
        mask = config.sample_mask(names, DEVICE)
        assert not bool(mask[0, 0].item()), "rate-1 var must always be dropped"
        # only var_0 fires and max_masked_vars=0, so nothing else is dropped
        assert int((~mask).sum().item()) == 1


def test_rate_one_var_always_dropped():
    """A rate-1 var is always dropped regardless of the uniform count k."""
    config = VariableMaskingConfig(
        max_masked_vars=2, variable_masking_rates={"var_0": 1.0}
    )
    names = _names(5)
    for _ in range(128):
        mask = config.sample_mask(names, DEVICE)
        assert not bool(mask[0, 0].item()), "rate-1 var must always be dropped"
        # k in [0, 2] plus the guaranteed var_0 (which may overlap the uniform set)
        assert 1 <= int((~mask).sum().item()) <= 3


def test_fired_var_in_uniform_set_does_not_double_count():
    """A fired var the uniform draw already dropped stays dropped (idempotent OR).

    With max_masked_vars == n_channels and a single rate-1 var, the dropped
    count never exceeds n_channels.
    """
    n = 4
    config = VariableMaskingConfig(
        max_masked_vars=n, variable_masking_rates={"var_0": 1.0}
    )
    names = _names(n)
    for _ in range(256):
        mask = config.sample_mask(names, DEVICE)
        n_masked = int((~mask).sum().item())
        assert 1 <= n_masked <= n
        assert not bool(mask[0, 0].item())


def test_multiple_fired_vars_all_dropped():
    """OR-combining: all fired vars are dropped, count >= number fired."""
    n = 5
    config = VariableMaskingConfig(
        max_masked_vars=1,
        variable_masking_rates={name: 1.0 for name in _names(n)},
    )
    names = _names(n)
    for _ in range(128):
        mask = config.sample_mask(names, DEVICE)
        # all n vars fire, so all are dropped regardless of k
        assert int((~mask).sum().item()) == n
