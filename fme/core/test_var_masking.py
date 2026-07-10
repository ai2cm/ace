import pytest
import torch

from fme.core.var_masking import (
    BernoulliMaskingConfig,
    MaskingGroupConfig,
    UniformMaskingConfig,
    VariableMaskingConfig,
)

DEVICE = torch.device("cpu")


def _names(n: int) -> list[str]:
    return [f"var_{i}" for i in range(n)]


def _group(variables: list[str], rate: float) -> MaskingGroupConfig:
    return MaskingGroupConfig(
        variables=variables, masking=BernoulliMaskingConfig(rate=rate)
    )


def test_max_masked_vars_validation():
    with pytest.raises(ValueError, match="max_masked_vars"):
        UniformMaskingConfig(max_masked_vars=-1)
    with pytest.raises(ValueError, match="max_masked_vars"):
        UniformMaskingConfig(max_masked_vars=True)


def test_group_rate_validation():
    with pytest.raises(ValueError, match="rate must be in"):
        BernoulliMaskingConfig(rate=-0.1)
    with pytest.raises(ValueError, match="rate must be in"):
        BernoulliMaskingConfig(rate=1.1)


def test_probability_validation():
    with pytest.raises(ValueError, match="probability must be in"):
        UniformMaskingConfig(max_masked_vars=1, probability=-0.1)
    with pytest.raises(ValueError, match="probability must be in"):
        UniformMaskingConfig(max_masked_vars=1, probability=1.1)
    # bool is a subclass of int; reject it rather than coerce to 0.0/1.0.
    with pytest.raises(ValueError, match="probability must be in"):
        UniformMaskingConfig(max_masked_vars=1, probability=True)
    with pytest.raises(ValueError, match="probability must be in"):
        UniformMaskingConfig(max_masked_vars=1, probability=False)


def test_empty_group_validation():
    with pytest.raises(ValueError, match="non-empty"):
        _group([], 0.5)


def test_duplicate_variable_across_groups_validation():
    with pytest.raises(ValueError, match="more than one masking group"):
        VariableMaskingConfig(
            override_groups=[_group(["a", "b"], 0.5), _group(["b"], 0.5)]
        )


def test_build_rejects_unknown_group_variable():
    config = VariableMaskingConfig(override_groups=[_group(["var_0", "typo"], 0.5)])
    with pytest.raises(ValueError, match="not in packed input channels"):
        config.build(_names(3))


def test_build_accepts_known_group_variables():
    config = VariableMaskingConfig(override_groups=[_group(["var_0", "var_2"], 0.5)])
    config.build(_names(3))


def test_build_noop_without_groups():
    VariableMaskingConfig(default=UniformMaskingConfig(2)).build(_names(3))


def _build(config: VariableMaskingConfig, n: int):
    return config.build(_names(n))


def test_sample_mask_shape_and_dtype():
    masking = _build(VariableMaskingConfig(default=UniformMaskingConfig(3)), 10)
    mask = masking.sample_mask(DEVICE)
    assert mask.shape == (1, 10)
    assert mask.dtype == torch.bool


def test_uniform_count_in_range():
    masking = _build(VariableMaskingConfig(default=UniformMaskingConfig(4)), 8)
    counts = set()
    for _ in range(256):
        mask = masking.sample_mask(DEVICE)
        n_masked = int((~mask).sum().item())
        assert 0 <= n_masked <= 4
        counts.add(n_masked)
    # min count is hard-coded 0, so zero-drop draws must be possible
    assert 0 in counts


def test_uniform_count_capped_at_n_channels():
    masking = _build(VariableMaskingConfig(default=UniformMaskingConfig(100)), 5)
    for _ in range(64):
        mask = masking.sample_mask(DEVICE)
        assert 0 <= int((~mask).sum().item()) <= 5


def test_no_masking_by_default():
    masking = _build(VariableMaskingConfig(), 5)
    for _ in range(16):
        assert masking.sample_mask(DEVICE).all()


def test_group_rate_zero_keeps_all():
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(0),
            override_groups=[_group(["var_0", "var_1"], 0.0)],
        ),
        4,
    )
    for _ in range(16):
        assert masking.sample_mask(DEVICE).all()


def test_rate_one_group_dropped_with_no_uniform_slots():
    """A rate-1 group is dropped solely by its Bernoulli, even with no uniform."""
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(0),
            override_groups=[_group(["var_0"], 1.0)],
        ),
        4,
    )
    for _ in range(32):
        mask = masking.sample_mask(DEVICE)
        assert not bool(mask[0, 0].item()), "rate-1 group must always be dropped"
        # only var_0 fires and max_masked_vars=0, so nothing else is dropped
        assert int((~mask).sum().item()) == 1


def test_rate_one_group_always_dropped():
    """A rate-1 group is always dropped regardless of the uniform count k."""
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(2),
            override_groups=[_group(["var_0"], 1.0)],
        ),
        5,
    )
    for _ in range(128):
        mask = masking.sample_mask(DEVICE)
        assert not bool(mask[0, 0].item()), "rate-1 group must always be dropped"
        # var_0 excluded from uniform pool of 4; dropped count is 1 plus k in [0, 2].
        assert 1 <= int((~mask).sum().item()) <= 3


def test_grouped_vars_excluded_from_uniform_pool():
    """Grouped channels are never dropped by the uniform mechanism.

    A rate-0 grouped var must stay present no matter how large max_masked_vars
    is, because it is excluded from the uniform pool entirely.
    """
    n = 4
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(n),
            override_groups=[_group(["var_0"], 0.0)],
        ),
        n,
    )
    for _ in range(256):
        mask = masking.sample_mask(DEVICE)
        assert bool(mask[0, 0].item()), "rate-0 grouped var must never be dropped"
        # uniform pool is var_1..var_3 (3 channels), so at most 3 are dropped
        assert 0 <= int((~mask).sum().item()) <= n - 1


def test_uniform_count_capped_at_pool_size():
    """The uniform count is capped at the number of ungrouped channels."""
    n = 5
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(100),
            override_groups=[_group(["var_0", "var_1"], 0.0)],
        ),
        n,
    )
    for _ in range(64):
        mask = masking.sample_mask(DEVICE)
        # var_0, var_1 are rate-0 and excluded; pool is the other 3 channels
        assert bool(mask[0, 0].item()) and bool(mask[0, 1].item())
        assert 0 <= int((~mask).sum().item()) <= 3


def test_marginal_rate_independent_of_uniform():
    """A group's drop frequency matches its rate, independent of uniform."""
    n = 6
    rate = 0.5
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(n),
            override_groups=[_group(["var_0"], rate)],
        ),
        n,
    )
    torch.manual_seed(0)
    trials = 4000
    dropped = sum(
        int(not bool(masking.sample_mask(DEVICE)[0, 0].item())) for _ in range(trials)
    )
    freq = dropped / trials
    assert abs(freq - rate) < 0.05, f"expected ~{rate}, got {freq}"


def test_group_members_dropped_together():
    """A fired multi-var group drops all its members jointly, never split."""
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(0),
            override_groups=[_group(["var_0", "var_1", "var_2"], 0.5)],
        ),
        4,
    )
    torch.manual_seed(0)
    saw_dropped = False
    saw_present = False
    for _ in range(256):
        mask = masking.sample_mask(DEVICE)
        members = [bool(mask[0, i].item()) for i in range(3)]
        # all three members share one draw: all present or all absent
        assert all(members) or not any(members), members
        # var_3 is ungrouped and max_masked_vars=0, so it is never dropped
        assert bool(mask[0, 3].item())
        saw_dropped |= not any(members)
        saw_present |= all(members)
    assert saw_dropped and saw_present


def test_masks_are_rank_independent(monkeypatch):
    """Two ranks with the same distributed seed draw distinct mask sequences.

    The per-rank RNG is seeded ``distributed_seed + rank``, so data-parallel
    ranks drop distinct channels from their distinct local batches. Guards the
    PR's data-parallel independence claim.
    """
    from fme.core.distributed import Distributed

    dist = Distributed.get_instance()

    def _sample_sequence(rank: int, trials: int) -> list[tuple[bool, ...]]:
        monkeypatch.setattr(type(dist), "rank", property(lambda self: rank))
        monkeypatch.setattr(dist, "get_seed", lambda: 0)
        masking = _build(VariableMaskingConfig(default=UniformMaskingConfig(4)), 8)
        return [
            tuple(bool(x) for x in masking.sample_mask(DEVICE)[0].tolist())
            for _ in range(trials)
        ]

    trials = 64
    seq0 = _sample_sequence(rank=0, trials=trials)
    seq1 = _sample_sequence(rank=1, trials=trials)
    assert seq0 != seq1


def test_masks_are_reproducible_for_fixed_seed_and_rank(monkeypatch):
    """Same seed + rank replays the same mask sequence (distributed-seeded)."""
    from fme.core.distributed import Distributed

    dist = Distributed.get_instance()
    monkeypatch.setattr(type(dist), "rank", property(lambda self: 3))
    monkeypatch.setattr(dist, "get_seed", lambda: 7)

    def _sample_sequence() -> list[tuple[bool, ...]]:
        masking = _build(VariableMaskingConfig(default=UniformMaskingConfig(4)), 8)
        return [
            tuple(bool(x) for x in masking.sample_mask(DEVICE)[0].tolist())
            for _ in range(32)
        ]

    assert _sample_sequence() == _sample_sequence()


def _sequence(config: VariableMaskingConfig, n: int, trials: int):
    masking = _build(config, n)
    return [
        tuple(bool(x) for x in masking.sample_mask(DEVICE)[0].tolist())
        for _ in range(trials)
    ]


def test_probability_one_matches_omitted():
    """probability=1.0 short-circuits the gate, leaving the RNG stream intact.

    An explicit probability=1.0 must replay exactly the same sequence as a config
    that omits the field, proving no gate draw is taken at 1.0 (so pre-field
    regression baselines are unchanged).
    """
    omitted = VariableMaskingConfig(default=UniformMaskingConfig(max_masked_vars=3))
    explicit = VariableMaskingConfig(
        default=UniformMaskingConfig(max_masked_vars=3, probability=1.0)
    )
    assert _sequence(omitted, 5, 64) == _sequence(explicit, 5, 64)


def test_probability_zero_never_drops():
    """probability=0.0 gates every step, so nothing is ever dropped."""
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(max_masked_vars=5, probability=0.0)
        ),
        5,
    )
    for _ in range(64):
        assert masking.sample_mask(DEVICE).all()


def test_probability_gates_uniform_marginal():
    """No-drop frequency matches ``(1 - p) + p / (n + 1)`` for a gated uniform.

    With ``max_masked_vars == n`` a fired scheme draws ``k`` uniform in
    ``[0, n]``, so it drops nothing with probability ``1 / (n + 1)``. Folding in
    the gate gives an overall no-drop frequency of ``(1 - p) + p / (n + 1)``.
    """
    n = 4
    probability = 0.5
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(max_masked_vars=n, probability=probability)
        ),
        n,
    )
    trials = 4000
    nodrop = sum(
        int(bool(masking.sample_mask(DEVICE).all().item())) for _ in range(trials)
    )
    freq = nodrop / trials
    expected = (1 - probability) + probability / (n + 1)
    assert abs(freq - expected) < 0.05, f"expected ~{expected}, got {freq}"


def test_uniform_probability_raises_nodrop_fraction():
    """Gating a uniform scheme increases the fraction of no-drop steps."""
    n = 6
    trials = 4000

    def _nodrop_fraction(probability: float) -> float:
        masking = _build(
            VariableMaskingConfig(
                default=UniformMaskingConfig(max_masked_vars=n, probability=probability)
            ),
            n,
        )
        nodrop = sum(
            int(bool(masking.sample_mask(DEVICE).all().item())) for _ in range(trials)
        )
        return nodrop / trials

    assert _nodrop_fraction(0.5) > _nodrop_fraction(1.0)


def test_probability_gates_override_group():
    """probability threads through override_groups, not just the default pool."""
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(max_masked_vars=0),
            override_groups=[
                MaskingGroupConfig(
                    variables=["var_0", "var_1"],
                    masking=UniformMaskingConfig(max_masked_vars=2, probability=0.0),
                )
            ],
        ),
        3,
    )
    # default masks nothing (max=0) and the group is gated off, so no channel
    # is ever dropped.
    for _ in range(64):
        assert masking.sample_mask(DEVICE).all()


def test_multiple_groups_fire_independently():
    """Distinct groups drop on their own draws; a fired group drops all members."""
    n = 5
    masking = _build(
        VariableMaskingConfig(
            default=UniformMaskingConfig(0),
            override_groups=[
                _group(["var_0", "var_1"], 1.0),
                _group(["var_2", "var_3"], 1.0),
            ],
        ),
        n,
    )
    for _ in range(128):
        mask = masking.sample_mask(DEVICE)
        # both rate-1 groups fire; all four grouped vars dropped, var_4 kept
        assert int((~mask).sum().item()) == 4
        assert bool(mask[0, 4].item())
