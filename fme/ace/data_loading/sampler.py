from collections.abc import Iterator, Sequence

import numpy as np
import torch

from fme.core.dataset.schedule import WeightSchedule
from fme.core.rand import alternate_seed


def build_group_ids(member_lengths: Sequence[int], groups: Sequence[int]) -> np.ndarray:
    """Map each concat sample to a group id.

    Args:
        member_lengths: Number of samples in each concat member dataset.
        groups: Number of consecutive concat members in each group. Must be
            positive and sum to ``len(member_lengths)``.

    Returns:
        Array of length ``sum(member_lengths)`` giving the group id of each
        sample, in concat order.
    """
    if any(g <= 0 for g in groups):
        raise ValueError(f"All group sizes must be positive, got {list(groups)}")
    if sum(groups) != len(member_lengths):
        raise ValueError(
            f"Sum of group sizes ({sum(groups)}) must equal the number of "
            f"concat members ({len(member_lengths)})"
        )
    group_of_member = np.repeat(np.arange(len(groups)), groups)
    return np.repeat(group_of_member, member_lengths)


class ScheduledWeightedSampler(torch.utils.data.Sampler):
    """Weighted, with-replacement, epoch-aware, per-rank sampler.

    Each sample is assigned a group, and groups are sampled according to a
    scheduled per-group weight. Within a group, samples are drawn uniformly.
    Different data-parallel ranks draw independent samples; the same base seed
    reproduces the same stream.
    """

    def __init__(
        self,
        sample_group_ids: np.ndarray,
        schedule: WeightSchedule,
        num_samples_per_rank: int,
        rank: int,
        base_seed: int,
        epoch: int = 0,
    ):
        self._group_ids = torch.as_tensor(sample_group_ids, dtype=torch.long)
        self._n_groups = len(schedule.start_value)
        self._group_counts = torch.bincount(self._group_ids, minlength=self._n_groups)
        self._schedule = schedule
        self._num_samples_per_rank = num_samples_per_rank
        self._rank = rank
        self._base_seed = base_seed
        self._draw_seed = 0
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._draw_seed = epoch
        group_weights = self._schedule.get_value(epoch)
        per_sample_weights = torch.zeros(len(self._group_ids), dtype=torch.double)
        for group in range(self._n_groups):
            count = int(self._group_counts[group].item())
            if count == 0:
                if group_weights[group] > 0:
                    raise ValueError(
                        f"Group {group} has positive weight "
                        f"{group_weights[group]} at epoch {epoch} but contains "
                        "no samples. Set its weight to 0 or remove the group."
                    )
                continue
            mask = self._group_ids == group
            per_sample_weights[mask] = group_weights[group] / count
        self._weights = per_sample_weights

    def alternate_shuffle(self):
        """Use an independent draw order without changing the scheduled epoch."""
        self._draw_seed = alternate_seed(self._epoch)

    def __len__(self) -> int:
        return self._num_samples_per_rank

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(
            self._base_seed + self._draw_seed * 1_000_003 + self._rank
        )
        indices = torch.multinomial(
            self._weights,
            self._num_samples_per_rank,
            replacement=True,
            generator=generator,
        )
        return iter(indices.tolist())
