import dataclasses
import logging
import math
from collections.abc import Iterator, Mapping, Sequence

import torch

from fme.core.dataset.dataset import DatasetABC
from fme.core.distributed import Distributed


@dataclasses.dataclass
class LabelSamplingConfig:
    """
    Configuration for drawing a fixed fraction of each epoch's training
    samples from each dataset label.

    Each epoch, every label gets a quota of samples proportional to its
    weight. Within a label, samples are drawn uniformly without replacement,
    reshuffling and starting over only once all of that label's samples have
    been drawn, so a label repeats samples only when its quota exceeds its
    number of samples. Quotas from all labels are shuffled together, so
    batches mix labels.

    Parameters:
        weights: Relative sampling weight for each label, normalized to sum
            to 1. Must name every label in the dataset; a weight of 0 excludes
            that label's samples from training.
        samples_per_epoch: Number of samples drawn per epoch across all
            data-parallel ranks. Defaults to the number of samples in the
            dataset.

    Note:
        Each sample must carry exactly one label. When ``time_buffer > 0``,
        a sample is a pre-loaded window that yields ``time_buffer + 1``
        consecutive batches, so label fractions hold on average over batches
        rather than within each batch.
    """

    weights: dict[str, float]
    samples_per_epoch: int | None = None

    def __post_init__(self):
        negative = {k: v for k, v in self.weights.items() if v < 0}
        if negative:
            raise ValueError(
                f"label_sampling weights must be non-negative, got {negative}"
            )
        if sum(self.weights.values()) <= 0:
            raise ValueError(
                "label_sampling weights must include at least one positive "
                f"weight, got {self.weights}"
            )
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError(
                "label_sampling samples_per_epoch must be positive, "
                f"got {self.samples_per_epoch}"
            )

    def build_sampler(self, dataset: DatasetABC) -> "LabelBalancedSampler":
        """
        Build a sampler for the given dataset on this data-parallel rank.

        Raises:
            ValueError: If the dataset's labels do not match the weight keys,
                if any sample does not carry exactly one label, or if a label
                with a positive weight has no samples.
        """
        dist = Distributed.get_instance()
        return LabelBalancedSampler(
            dataset,
            sample_labels=_get_single_labels(dataset, set(self.weights)),
            weights=self.weights,
            samples_per_epoch=self.samples_per_epoch,
            num_replicas=dist.total_data_parallel_ranks,
            rank=dist.data_parallel_rank,
            seed=dist.get_seed(),
        )


def _get_single_labels(dataset: DatasetABC, expected: set[str]) -> list[str]:
    available: set[str] = set()
    single_labels = []
    for i, labels in enumerate(dataset.sample_labels):
        if labels is None or len(labels) != 1:
            raise ValueError(
                "label_sampling requires every sample to carry exactly one "
                f"label, but sample {i} has labels {labels}"
            )
        (label,) = labels
        available.add(label)
        single_labels.append(label)
    if available != expected:
        raise ValueError(
            "label_sampling weights must name exactly the dataset's labels, "
            f"got weights for {sorted(expected)} but dataset labels "
            f"{sorted(available)}"
        )
    return single_labels


def _get_quotas(weights: Mapping[str, float], total: int) -> dict[str, int]:
    """
    Split ``total`` samples across labels in proportion to ``weights`` by
    largest-remainder rounding, giving each positively-weighted label at
    least one sample.
    """
    positive = [label for label, w in weights.items() if w > 0]
    if total < len(positive):
        raise ValueError(
            f"Cannot draw at least one sample for each of {len(positive)} "
            f"positively-weighted labels with only {total} samples per epoch"
        )
    weight_sum = sum(weights.values())
    exact = {label: weights[label] / weight_sum * total for label in positive}
    quotas = {label: 0 for label in weights}
    for label in positive:
        quotas[label] = max(math.floor(exact[label]), 1)
    by_remainder = sorted(
        positive, key=lambda label: exact[label] - quotas[label], reverse=True
    )
    shortfall = total - sum(quotas.values())
    for label in by_remainder[:shortfall]:
        quotas[label] += 1
    # Bumping labels up to 1 sample can overshoot the total; take the excess
    # from the labels furthest above their exact share.
    while sum(quotas.values()) > total:
        label = max(
            (label for label in positive if quotas[label] > 1),
            key=lambda label: quotas[label] - exact[label],
        )
        quotas[label] -= 1
    return quotas


def _draw_cycling(
    indices: torch.Tensor, n: int, generator: torch.Generator
) -> torch.Tensor:
    """
    Draw ``n`` of ``indices`` by concatenating fresh shuffles of them, so no
    index repeats until every index has been drawn.
    """
    n_shuffles = math.ceil(n / len(indices))
    shuffles = [
        indices[torch.randperm(len(indices), generator=generator)]
        for _ in range(n_shuffles)
    ]
    return torch.cat(shuffles)[:n]


class LabelBalancedSampler(torch.utils.data.DistributedSampler):
    """
    Samples a fixed number of dataset indices per label each epoch, split
    across data-parallel ranks.

    Every rank builds the same global order from ``seed + epoch`` and takes
    every ``num_replicas``-th index starting at its ``rank``, so spatial
    co-ranks (which share a data-parallel rank) receive identical indices.
    Subclasses DistributedSampler so ``set_epoch`` reshuffles it as for the
    default training sampler.
    """

    def __init__(
        self,
        dataset: DatasetABC,
        sample_labels: Sequence[str],
        weights: Mapping[str, float],
        samples_per_epoch: int | None,
        num_replicas: int,
        rank: int,
        seed: int,
    ):
        """
        Args:
            dataset: Dataset to sample from.
            sample_labels: Label of each sample in the dataset.
            weights: Relative sampling weight for each label.
            samples_per_epoch: Samples drawn per epoch across all ranks;
                defaults to ``len(sample_labels)``. Rounded up to a multiple
                of ``num_replicas`` so every rank draws the same number.
            num_replicas: Number of data-parallel ranks.
            rank: Data-parallel rank of this process.
            seed: Seed shared by all ranks.
        """
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed
        )
        if len(sample_labels) != len(dataset):
            raise ValueError(
                f"Got {len(sample_labels)} sample labels for a dataset of "
                f"{len(dataset)} samples"
            )
        n_requested = (
            samples_per_epoch if samples_per_epoch is not None else len(sample_labels)
        )
        self.num_samples = math.ceil(n_requested / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self._quotas = _get_quotas(weights, self.total_size)
        indices_by_label: dict[str, list[int]] = {label: [] for label in weights}
        for i, label in enumerate(sample_labels):
            indices_by_label[label].append(i)
        self._indices_by_label = {
            label: torch.tensor(indices, dtype=torch.long)
            for label, indices in indices_by_label.items()
        }
        empty = [
            label
            for label, quota in self._quotas.items()
            if quota > 0 and len(self._indices_by_label[label]) == 0
        ]
        if empty:
            raise ValueError(
                f"label_sampling labels {empty} have positive weight but no samples"
            )
        for label, quota in self._quotas.items():
            n_available = len(self._indices_by_label[label])
            logging.info(
                f"label_sampling: label {label!r} draws {quota} of "
                f"{self.total_size} samples per epoch "
                f"({quota / self.total_size:.1%}) from {n_available} samples "
                f"(each seen {quota / max(n_available, 1):.2f}x per epoch)"
            )

    @property
    def quotas(self) -> dict[str, int]:
        """Number of samples drawn from each label per epoch, across all ranks."""
        return dict(self._quotas)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        drawn = torch.cat(
            [
                _draw_cycling(self._indices_by_label[label], quota, generator)
                for label, quota in self._quotas.items()
                if quota > 0
            ]
        )
        order = drawn[torch.randperm(self.total_size, generator=generator)]
        return iter(order[self.rank : self.total_size : self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples
