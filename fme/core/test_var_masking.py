import pytest
import torch

from fme.core.var_masking import VariableMaskingConfig

DEVICE = torch.device("cpu")


def _names(n: int) -> list[str]:
    return [f"var_{i}" for i in range(n)]


def test_max_masked_vars_validation():
    with pytest.raises(ValueError, match="max_masked_vars"):
        VariableMaskingConfig(max_masked_vars=-1)
    with pytest.raises(ValueError, match="max_masked_vars"):
        VariableMaskingConfig(max_masked_vars=True)


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
    """A rate-1 var is dropped solely by its Bernoulli, even with no uniform."""
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
        # var_0 (named) is excluded from the uniform pool of 4, so the dropped
        # count is exactly 1 (guaranteed var_0) plus k in [0, 2].
        assert 1 <= int((~mask).sum().item()) <= 3


def test_named_vars_excluded_from_uniform_pool():
    """Named channels are never dropped by the uniform mechanism.

    A rate-0 named var must stay present no matter how large max_masked_vars
    is, because it is excluded from the uniform pool entirely.
    """
    n = 4
    config = VariableMaskingConfig(
        max_masked_vars=n, variable_masking_rates={"var_0": 0.0}
    )
    names = _names(n)
    for _ in range(256):
        mask = config.sample_mask(names, DEVICE)
        assert bool(mask[0, 0].item()), "rate-0 named var must never be dropped"
        # uniform pool is var_1..var_3 (3 channels), so at most 3 are dropped
        assert 0 <= int((~mask).sum().item()) <= n - 1


def test_uniform_count_capped_at_pool_size():
    """The uniform count is capped at the number of unnamed channels."""
    n = 5
    config = VariableMaskingConfig(
        max_masked_vars=100,
        variable_masking_rates={"var_0": 0.0, "var_1": 0.0},
    )
    names = _names(n)
    for _ in range(64):
        mask = config.sample_mask(names, DEVICE)
        # var_0, var_1 are rate-0 and excluded; pool is the other 3 channels
        assert bool(mask[0, 0].item()) and bool(mask[0, 1].item())
        assert 0 <= int((~mask).sum().item()) <= 3


def test_marginal_rate_independent_of_uniform():
    """A named var's drop frequency matches its rate, independent of uniform."""
    n = 6
    rate = 0.5
    config = VariableMaskingConfig(
        max_masked_vars=n, variable_masking_rates={"var_0": rate}
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


def test_multiple_fired_vars_all_dropped():
    """All fired vars are dropped; with all channels named, count == n."""
    n = 5
    config = VariableMaskingConfig(
        max_masked_vars=1,
        variable_masking_rates={name: 1.0 for name in _names(n)},
    )
    names = _names(n)
    for _ in range(128):
        mask = config.sample_mask(names, DEVICE)
        # all n vars fire (and the uniform pool is empty), so all are dropped
        assert int((~mask).sum().item()) == n
