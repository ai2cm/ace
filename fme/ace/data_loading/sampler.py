from collections.abc import Iterator, Sequence

import numpy as np
import torch

from fme.core.dataset.schedule import WeightSchedule
from fme.core.distributed import Distributed
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


def build_member_ids(member_lengths: Sequence[int]) -> np.ndarray:
    """Map each concat sample to its concat member index.

    Args:
        member_lengths: Number of samples in each concat member dataset.

    Returns:
        Array of length ``sum(member_lengths)`` giving the member index of each
        sample, in concat order.
    """
    return np.repeat(np.arange(len(member_lengths)), member_lengths)


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
        sample_member_ids: np.ndarray | None = None,
        member_labels: Sequence[str] | None = None,
        group_labels: Sequence[str] | None = None,
    ):
        self._group_ids = torch.as_tensor(sample_group_ids, dtype=torch.long)
        self._n_groups = len(schedule.start_value)
        self._group_counts = torch.bincount(self._group_ids, minlength=self._n_groups)
        self._schedule = schedule
        self._num_samples_per_rank = num_samples_per_rank
        self._rank = rank
        self._base_seed = base_seed
        self._draw_seed = 0
        if sample_member_ids is None:
            sample_member_ids = sample_group_ids
        self._member_ids = torch.as_tensor(sample_member_ids, dtype=torch.long)
        if len(self._member_ids) != len(self._group_ids):
            raise ValueError(
                "sample_member_ids and sample_group_ids must have the same length, "
                f"got {len(self._member_ids)} and {len(self._group_ids)}"
            )
        self._n_members = int(self._member_ids.max().item()) + 1
        if member_labels is None:
            member_labels = [str(i) for i in range(self._n_members)]
        if len(member_labels) != self._n_members:
            raise ValueError(
                f"member_labels must have one entry per member ({self._n_members}), "
                f"got {len(member_labels)}"
            )
        self._member_labels = list(member_labels)
        if group_labels is None:
            group_labels = [str(i) for i in range(self._n_groups)]
        if len(group_labels) != self._n_groups:
            raise ValueError(
                f"group_labels must have one entry per group ({self._n_groups}), "
                f"got {len(group_labels)}"
            )
        self._group_labels = list(group_labels)
        # realized per-group / per-member draw counts from the most recent
        # __iter__; populated lazily as the sampler is iterated each epoch.
        self._last_group_counts: torch.Tensor | None = None
        self._last_member_counts: torch.Tensor | None = None
        self._last_total = 0
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
        # multinomial draws the whole epoch up front, so these counts reflect the
        # full draw regardless of how many indices the consumer actually pulls.
        self._last_group_counts = torch.bincount(
            self._group_ids[indices], minlength=self._n_groups
        )
        self._last_member_counts = torch.bincount(
            self._member_ids[indices], minlength=self._n_members
        )
        self._last_total = len(indices)
        return iter(indices.tolist())

    def get_realized_fractions(self) -> dict[str, float]:
        """Fraction of samples drawn from each group and member in the last draw.

        Counts are summed across data-parallel ranks (each rank draws an
        independent stream) so the fractions reflect the whole batch of samples
        the model saw, not just the local rank's. Returns an empty dict if the
        sampler has not been iterated yet. Keys are namespaced
        ``group/<label>`` and ``member/<label>``; values sum to 1 within each
        namespace.

        Note: this performs collective all-reduces, so every rank must call it.
        """
        if (
            self._last_total == 0
            or self._last_group_counts is None
            or self._last_member_counts is None
        ):
            return {}
        # group_weights forbids spatial parallelism, so an all-reduce over the
        # world sums exactly over the data-parallel ranks. clone() because
        # reduce_sum mutates its input in place.
        dist = Distributed.get_instance()
        group_counts = dist.reduce_sum(self._last_group_counts.clone().double())
        member_counts = dist.reduce_sum(self._last_member_counts.clone().double())
        total = dist.reduce_sum(
            torch.tensor(float(self._last_total), dtype=torch.double)
        ).item()
        logs: dict[str, float] = {}
        for group in range(self._n_groups):
            logs[f"group/{self._group_labels[group]}"] = (
                group_counts[group].item() / total
            )
        for member in range(self._n_members):
            logs[f"member/{self._member_labels[member]}"] = (
                member_counts[member].item() / total
            )
        return logs
