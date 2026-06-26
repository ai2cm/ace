import numpy as np
import pytest
import torch

from fme.ace.data_loading.sampler import (
    ScheduledWeightedSampler,
    build_group_ids,
    build_member_ids,
)
from fme.core.dataset.schedule import WeightMilestone, WeightSchedule


def test_build_group_ids():
    # 4 members with lengths 2, 3, 1, 4 split into groups [1, 2, 1]
    gids = build_group_ids([2, 3, 1, 4], [1, 2, 1])
    # group 0 = member0 (2), group 1 = members1,2 (3+1=4), group 2 = member3 (4)
    expected = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(gids, expected)


def test_build_group_ids_non_positive_raises():
    with pytest.raises(ValueError, match="positive"):
        build_group_ids([2, 3], [0, 2])


def test_build_group_ids_sum_mismatch_raises():
    with pytest.raises(ValueError, match="must equal the number of concat members"):
        build_group_ids([2, 3, 1], [1, 1])


def _counts_by_group(indices, gids, n_groups):
    group_of_index = gids[np.asarray(indices)]
    return np.bincount(group_of_index, minlength=n_groups)


def test_sampler_draws_in_expected_proportions():
    gids = build_group_ids([100, 100, 100], [1, 1, 1])
    schedule = WeightSchedule.from_constant([0.2, 0.3, 0.5])
    sampler = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=30_000, rank=0, base_seed=0
    )
    indices = list(sampler)
    counts = _counts_by_group(indices, gids, 3)
    fractions = counts / counts.sum()
    np.testing.assert_allclose(fractions, [0.2, 0.3, 0.5], atol=0.02)


def test_sampler_set_epoch_switches_proportions():
    gids = build_group_ids([100, 100], [1, 1])
    schedule = WeightSchedule(
        start_value=[0.5, 0.5],
        milestones=[WeightMilestone(epoch=2, value=[0.9, 0.1])],
    )
    sampler = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=20_000, rank=0, base_seed=0
    )
    before = _counts_by_group(list(sampler), gids, 2)
    np.testing.assert_allclose(before / before.sum(), [0.5, 0.5], atol=0.02)

    sampler.set_epoch(2)
    after = _counts_by_group(list(sampler), gids, 2)
    np.testing.assert_allclose(after / after.sum(), [0.9, 0.1], atol=0.02)


def test_sampler_zeroed_group_never_drawn():
    gids = build_group_ids([50, 50], [1, 1])
    schedule = WeightSchedule.from_constant([0.0, 1.0])
    sampler = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=5_000, rank=0, base_seed=0
    )
    indices = np.array(list(sampler))
    assert (gids[indices] == 0).sum() == 0


def test_sampler_same_base_seed_reproducible():
    gids = build_group_ids([20, 20], [1, 1])
    schedule = WeightSchedule.from_constant([0.5, 0.5])
    a = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=100, rank=0, base_seed=7
    )
    b = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=100, rank=0, base_seed=7
    )
    assert list(a) == list(b)


def test_sampler_different_base_seed_changes_draw():
    gids = build_group_ids([20, 20], [1, 1])
    schedule = WeightSchedule.from_constant([0.5, 0.5])
    a = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=100, rank=0, base_seed=7
    )
    b = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=100, rank=0, base_seed=8
    )
    assert list(a) != list(b)


def test_sampler_ranks_draw_different_samples():
    gids = build_group_ids([20, 20], [1, 1])
    schedule = WeightSchedule.from_constant([0.5, 0.5])
    rank0 = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=100, rank=0, base_seed=7
    )
    rank1 = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=100, rank=1, base_seed=7
    )
    assert list(rank0) != list(rank1)


def test_sampler_alternate_shuffle_changes_order_not_weights():
    gids = build_group_ids([100, 100], [1, 1])
    schedule = WeightSchedule.from_constant([0.5, 0.5])
    sampler = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=10_000, rank=0, base_seed=0
    )
    weights_before = sampler._weights.clone()
    first = list(sampler)
    sampler.alternate_shuffle()
    second = list(sampler)
    assert first != second
    torch.testing.assert_close(weights_before, sampler._weights)


def test_sampler_len():
    gids = build_group_ids([10, 10], [1, 1])
    schedule = WeightSchedule.from_constant([0.5, 0.5])
    sampler = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=42, rank=0, base_seed=0
    )
    assert len(sampler) == 42


def test_sampler_positive_weight_on_empty_group_raises():
    # group 1 has no samples because member lengths put everything in group 0
    gids = build_group_ids([10, 0], [1, 1])
    schedule = WeightSchedule.from_constant([0.5, 0.5])
    with pytest.raises(ValueError, match="no samples"):
        ScheduledWeightedSampler(
            gids, schedule, num_samples_per_rank=10, rank=0, base_seed=0
        )


def test_sampler_zero_weight_on_empty_group_allowed():
    gids = build_group_ids([10, 0], [1, 1])
    schedule = WeightSchedule.from_constant([1.0, 0.0])
    sampler = ScheduledWeightedSampler(
        gids, schedule, num_samples_per_rank=10, rank=0, base_seed=0
    )
    assert len(list(sampler)) == 10


def test_build_member_ids():
    np.testing.assert_array_equal(
        build_member_ids([2, 3, 1]), np.array([0, 0, 1, 1, 1, 2])
    )


def _make_member_weighted_sampler(num_samples_per_rank):
    member_lengths = [100, 100, 100]
    gids = build_group_ids(member_lengths, [2, 1])  # members 0,1 -> group0; 2 -> group1
    mids = build_member_ids(member_lengths)
    schedule = WeightSchedule.from_constant([0.6, 0.4])
    return ScheduledWeightedSampler(
        gids,
        schedule,
        num_samples_per_rank=num_samples_per_rank,
        rank=0,
        base_seed=0,
        sample_member_ids=mids,
        member_labels=["amip", "co2", "era5"],
        group_labels=["other", "era5"],
    )


def test_realized_fractions_empty_before_iteration():
    sampler = _make_member_weighted_sampler(100)
    assert sampler.get_realized_fractions() == {}


def test_realized_fractions_match_weights():
    sampler = _make_member_weighted_sampler(40_000)
    list(sampler)  # trigger the draw
    fractions = sampler.get_realized_fractions()
    # groups normalized from [0.6, 0.4]
    assert fractions["group/other"] == pytest.approx(0.6, abs=0.02)
    assert fractions["group/era5"] == pytest.approx(0.4, abs=0.02)
    # within group "other" the two equal-size members split the 0.6 evenly
    assert fractions["member/amip"] == pytest.approx(0.3, abs=0.02)
    assert fractions["member/co2"] == pytest.approx(0.3, abs=0.02)
    assert fractions["member/era5"] == pytest.approx(0.4, abs=0.02)
    assert sum(v for k, v in fractions.items() if k.startswith("group/")) == (
        pytest.approx(1.0)
    )
    assert sum(v for k, v in fractions.items() if k.startswith("member/")) == (
        pytest.approx(1.0)
    )


def test_realized_fractions_cooldown_zeroes_other_members():
    member_lengths = [50, 50]
    gids = build_group_ids(member_lengths, [1, 1])
    mids = build_member_ids(member_lengths)
    schedule = WeightSchedule.from_constant([0.0, 1.0])
    sampler = ScheduledWeightedSampler(
        gids,
        schedule,
        num_samples_per_rank=2_000,
        rank=0,
        base_seed=0,
        sample_member_ids=mids,
        member_labels=["other", "era5"],
        group_labels=["other", "era5"],
    )
    list(sampler)
    fractions = sampler.get_realized_fractions()
    assert fractions["member/other"] == 0.0
    assert fractions["member/era5"] == 1.0


def test_member_labels_length_mismatch_raises():
    member_lengths = [10, 10]
    gids = build_group_ids(member_lengths, [1, 1])
    mids = build_member_ids(member_lengths)
    schedule = WeightSchedule.from_constant([0.5, 0.5])
    with pytest.raises(ValueError, match="member_labels"):
        ScheduledWeightedSampler(
            gids,
            schedule,
            num_samples_per_rank=10,
            rank=0,
            base_seed=0,
            sample_member_ids=mids,
            member_labels=["only-one"],
        )
