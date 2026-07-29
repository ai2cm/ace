"""The multi-stream training/validation data container.

:class:`TranslateGriddedData` is a :class:`GriddedDataABC` over
:class:`TranslateBatchData`. It holds one loader per pairing group and zips
them: each group is shuffled independently, and one batch carries every stream.
It also exposes ``dataset_info`` keyed by *domain* — the pairing key
``ComponentPoolConfig.build`` consumes — which is not part of the ABC but is the
convention every ace package's training entry point follows.
"""

import itertools
import logging
from collections.abc import Iterator, Mapping, Sequence

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset_info import DatasetInfo, IncompatibleDatasetInfo
from fme.core.generics.data import DataLoader, GriddedDataABC, SizedMap
from fme.translate.data.batch_data import TranslateBatchData
from fme.translate.data.dataloader import TranslateDataLoader

__all__ = ["TranslateGriddedData"]


class ZippedGroupLoader:
    """Zips the pairing groups' loaders into one loader over all streams.

    Iteration stops with the shortest group, so ``__len__`` is the minimum over
    the groups: a group with fewer batches would otherwise have to be re-drawn
    mid-epoch, correlating its samples within the epoch.
    """

    def __init__(self, loaders: Sequence[DataLoader[TranslateBatchData]]):
        if not loaders:
            raise ValueError("At least one pairing group's loader is required.")
        self._loaders = list(loaders)

    def __len__(self) -> int:
        return min(len(loader) for loader in self._loaders)

    def __iter__(self) -> Iterator[TranslateBatchData]:
        for group_batches in zip(*self._loaders):
            yield TranslateBatchData.merge(group_batches)


def _dataset_info(properties: DatasetProperties) -> DatasetInfo:
    return DatasetInfo(
        horizontal_coordinates=properties.horizontal_coordinates,
        vertical_coordinate=properties.vertical_coordinate,
        spatial_mask_provider=properties.spatial_mask_provider,
        timestep=properties.timestep,
        variable_metadata=properties.variable_metadata,
        all_labels=properties.all_labels,
    )


def _domain_dataset_info(
    domain: str, entries: Sequence[tuple[str, DatasetProperties]]
) -> DatasetInfo:
    """One ``DatasetInfo`` for a domain, from the streams that serve it.

    Streams sharing a domain must describe compatible datasets (same grid,
    vertical coordinate, mask and timestep), checked with
    ``DatasetInfo.assert_compatible_with`` over every *ordered* pair. Both
    aspects of that matter:

    - every pair, not each stream against the first one, because
      ``assert_compatible_with`` skips a comparison either side declines to make
      (a ``NullVerticalCoordinate`` opts out of the vertical check), so a stream
      can be compatible with the first while disagreeing with a third;
    - both directions, because the timestep comparison is guarded on the
      *caller's* timestep being set. Checking one way only would let a stream
      with no inferred timestep silently pass against one that has it — and,
      since the returned info takes its coordinates from ``entries[0]``, publish
      that ``None`` for the whole domain.

    Only after those checks do the shared fields of ``entries[0]`` describe every
    stream. Variable metadata and labels are unioned; on a key present in two
    streams with conflicting metadata the later one wins and
    ``assert_compatible_with`` logs a warning, matching how ace merges metadata
    across datasets.
    """
    infos = [(name, _dataset_info(properties)) for name, properties in entries]
    for (name, info), (other_name, other) in itertools.permutations(infos, 2):
        try:
            info.assert_compatible_with(other)
        except IncompatibleDatasetInfo as err:
            raise ValueError(
                f"Streams {name!r} and {other_name!r} both serve domain "
                f"{domain!r} but describe incompatible datasets: {err}"
            ) from err
    reference = entries[0][1]
    variable_metadata: dict[str, VariableMetadata] = {}
    all_labels: set[str] | None = None
    for _, properties in entries:
        variable_metadata.update(properties.variable_metadata)
        if properties.all_labels is not None:
            all_labels = set(properties.all_labels) | (all_labels or set())
    return DatasetInfo(
        horizontal_coordinates=reference.horizontal_coordinates,
        vertical_coordinate=reference.vertical_coordinate,
        spatial_mask_provider=reference.spatial_mask_provider,
        timestep=reference.timestep,
        variable_metadata=variable_metadata,
        all_labels=all_labels,
    )


class TranslateGriddedData(GriddedDataABC[TranslateBatchData]):
    """Multi-stream data as required for translate training.

    All data exposed from this class is on the current device.

    Semantics under unequal pairing groups: ``n_batches`` is the minimum over
    the groups, and ``n_samples`` is ``n_batches * batch_size`` — that is, the
    number of samples drawn *per group*, not summed across groups, so it stays
    the count a per-batch metric divides by. The groups are shuffled
    independently: each group's sampler is built with its own ``seed_offset``,
    without which two equal-length groups would be handed the same permutation
    and so be index-paired in fact while claiming to be independent.

    Batches are moved to the device but not spatially scattered, following
    ``fme.coupled``: spatial model parallelism would need each stream scattered
    against its own grid, which no translate module supports yet. Data-parallel
    sharding is handled, by the per-group distributed samplers.
    """

    def __init__(
        self,
        group_loaders: Sequence[TranslateDataLoader],
        stream_properties: Mapping[str, DatasetProperties],
        stream_domains: Mapping[str, str],
    ):
        """
        Args:
            group_loaders: One loader per pairing group. They must share a batch
                size, and their streams must partition ``stream_properties`` —
                not checked here, since a loader does not expose its streams;
                :func:`fme.translate.data.getters.get_gridded_data` gets it from
                ``TranslateDataRequirements``, whose groups are validated to
                partition its streams.
            stream_properties: Each stream's dataset properties. Data can be on
                any device.
            stream_domains: The domain each stream serves.
        """
        if not group_loaders:
            raise ValueError("At least one pairing group's loader is required.")
        if set(stream_properties) != set(stream_domains):
            raise ValueError(
                "stream_properties and stream_domains must cover the same "
                f"streams, got {sorted(stream_properties)} and "
                f"{sorted(stream_domains)}."
            )
        batch_sizes = {loader.batch_size for loader in group_loaders}
        if len(batch_sizes) != 1:
            raise ValueError(
                f"All pairing groups must share a batch size, got {batch_sizes}."
            )
        self._group_loaders = list(group_loaders)
        self._batch_size = batch_sizes.pop()
        self._properties = {
            name: properties.to_device()
            for name, properties in stream_properties.items()
        }
        self._stream_domains = dict(stream_domains)
        domain_entries: dict[str, list[tuple[str, DatasetProperties]]] = {}
        for name, properties in self._properties.items():
            domain_entries.setdefault(self._stream_domains[name], []).append(
                (name, properties)
            )
        self._dataset_info = {
            domain: _domain_dataset_info(domain, entries)
            for domain, entries in domain_entries.items()
        }

    @property
    def dataset_info(self) -> Mapping[str, DatasetInfo]:
        """Per-domain dataset information, keyed by domain name.

        One entry per data-backed domain: this is what
        ``ComponentPoolConfig.build`` consumes to pair components with data.
        """
        return self._dataset_info

    @property
    def stream_domains(self) -> Mapping[str, str]:
        """The domain each stream serves."""
        return self._stream_domains

    @property
    def variable_metadata(self) -> Mapping[str, Mapping[str, VariableMetadata]]:
        """Each stream's variable metadata, keyed by stream name.

        Kept per stream rather than flattened: two streams at different
        resolutions carry the same variable names, so a flat merge would hide
        one stream's metadata behind another's.
        """
        return {
            name: properties.variable_metadata
            for name, properties in self._properties.items()
        }

    @property
    def loader(self) -> DataLoader[TranslateBatchData]:
        return self._on_device(ZippedGroupLoader(self._group_loaders))

    def subset_loader(
        self, start_batch: int | None = None, stop_batch: int | None = None
    ) -> DataLoader[TranslateBatchData]:
        return self._on_device(
            ZippedGroupLoader(
                [
                    loader.subset(start_batch=start_batch, stop_batch=stop_batch)
                    for loader in self._group_loaders
                ]
            )
        )

    def _on_device(
        self, base_loader: DataLoader[TranslateBatchData]
    ) -> DataLoader[TranslateBatchData]:
        return SizedMap(lambda batch: batch.to_device(), base_loader)

    @property
    def n_samples(self) -> int:
        return self.n_batches * self.batch_size

    @property
    def n_batches(self) -> int:
        return min(len(loader) for loader in self._group_loaders)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def set_epoch(self, epoch: int):
        """Set the epoch on every group's dataset and sampler."""
        for loader in self._group_loaders:
            loader.set_epoch(epoch)

    def alternate_shuffle(self):
        """Change every group's random shuffle for the current epoch."""
        for loader in self._group_loaders:
            loader.alternate_shuffle()

    def log_info(self, name: str):
        logging.info(
            f"{name} data: {self.n_batches} batches of {self.batch_size} samples "
            f"per stream, over {len(self._group_loaders)} independently sampled "
            f"pairing group(s) covering {len(self._properties)} stream(s)."
        )
        for group_index, loader in enumerate(self._group_loaders):
            loader.log_info(f"{name} pairing group {group_index}")
