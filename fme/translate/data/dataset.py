"""An N-stream dataset whose streams share one time index.

One :class:`PairedStreamDataset` backs one pairing group: index *i* yields one
sample from every stream in the group, all starting at the same valid time.

This is a translate-local class rather than a reuse of
:class:`fme.downscaling.data.datasets.FineCoarsePairedDataset`, which is
hard-wired to a fine/coarse pair and to single-timestep items (its
``BatchItemDatasetAdapter`` squeezes the time dimension away), so it cannot
carry the time windows a forward-prediction objective needs. The closest
in-repo analogue is :class:`fme.coupled.data_loading.data_typing.CoupledDataset`
— also index-paired, also validating alignment at construction — but likewise
fixed at two named components. What is reused instead is the machinery beneath
both: :class:`fme.core.generics.dataset.GenericDataset`, whose
:class:`~fme.core.generics.dataloader.GenericDataLoader` supplies subsetting,
epoch propagation and shuffle control.
"""

from collections.abc import Mapping

import numpy as np
import xarray as xr

from fme.core.dataset.dataset import DatasetABC, DatasetItem
from fme.core.generics.dataset import GenericDataset

__all__ = ["PairedStreamDataset"]


def _assert_start_times_match(
    reference_name: str,
    reference_times: xr.CFTimeIndex,
    name: str,
    times: xr.CFTimeIndex,
) -> None:
    """Raise unless two streams' sample start times agree index by index."""
    if reference_times.equals(times):
        return
    unequal = np.flatnonzero(np.asarray(reference_times) != np.asarray(times))
    if unequal.size > 0:
        index = int(unequal[0])
        detail = (
            f"at sample index {index}, {reference_name!r} starts at "
            f"{reference_times[index]} but {name!r} starts at {times[index]}"
        )
    else:
        detail = (
            f"their time indexes compare unequal: {reference_times!r} versus "
            f"{times!r}"
        )
    raise ValueError(
        f"Streams {reference_name!r} and {name!r} are sampled at the same valid "
        f"times because they co-occur in an objective, but {detail}. Align them, "
        "e.g. with a `subset:` on the stream that starts earlier."
    )


class PairedStreamDataset(GenericDataset[dict[str, DatasetItem]]):
    """Several data streams sampled at one shared time index.

    Sample *i* is ``{stream_name: sample_i}`` over every stream. Because the
    streams are paired by index, they must agree on their timestep and on their
    sample start times; both are checked here, at build time, with the
    offending stream pair named.

    The dataset's length is the minimum over its streams, since windows of
    different lengths (objectives asking for different ``n_timesteps``) leave
    different numbers of valid start times.
    """

    def __init__(self, datasets: Mapping[str, DatasetABC]):
        """
        Args:
            datasets: The group's per-stream datasets, keyed by stream name.
        """
        if not datasets:
            raise ValueError("A PairedStreamDataset requires at least one stream.")
        self._datasets = dict(datasets)
        self._length = min(len(dataset) for dataset in self._datasets.values())
        self._validate()

    def _validate(self) -> None:
        names = list(self._datasets)
        reference_name = names[0]
        reference = self._datasets[reference_name]
        reference_timestep = reference.properties.timestep
        reference_times = reference.sample_start_times[: self._length]
        for name in names[1:]:
            dataset = self._datasets[name]
            timestep = dataset.properties.timestep
            if timestep != reference_timestep:
                raise ValueError(
                    "Streams sampled at the same valid times must have the same "
                    f"timestep, but stream {reference_name!r} has timestep "
                    f"{reference_timestep} and stream {name!r} has timestep "
                    f"{timestep}. Pairing them by sample index would compare "
                    "different times. (Coarsening one stream in time, so the "
                    "two share a timestep, is the fix; an integer-ratio pairing "
                    "like fme.coupled's n_steps_fast is a deliberate "
                    "non-feature here until an objective needs it.)"
                )
            _assert_start_times_match(
                reference_name,
                reference_times,
                name,
                dataset.sample_start_times[: self._length],
            )

    def __getitem__(self, index) -> dict[str, DatasetItem]:
        return {name: dataset[index] for name, dataset in self._datasets.items()}

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        for dataset in self._datasets.values():
            dataset.set_epoch(epoch)

    def enable_shared_memory(self) -> None:
        for dataset in self._datasets.values():
            dataset.enable_shared_memory()

    @property
    def first_time(self):
        """Start time of the first sample, shared by every stream."""
        return next(iter(self._datasets.values())).sample_start_times[0]

    @property
    def last_time(self):
        """Start time of the last sample, shared by every stream."""
        return next(iter(self._datasets.values())).sample_start_times[self._length - 1]
