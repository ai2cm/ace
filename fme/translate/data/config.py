"""Configuration of the translate training/validation data: a list of streams.

Field names and defaults mirror :class:`fme.ace.data_loading.config.DataLoaderConfig`
so the ``train_data:`` block reads like ace's. What it adds is the ``streams``
list, and what it deliberately lacks is any sampling knob: which streams are
sampled together is derived from objective co-occurrence (see
:mod:`fme.translate.data.requirements`).
"""

import dataclasses

from fme.core.dataset.concat import ConcatDatasetConfig
from fme.core.dataset.dataset import DatasetABC
from fme.core.dataset.merged import MergeDatasetConfig
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.distributed import Distributed

__all__ = ["StreamConfig", "TranslateDataLoaderConfig"]


@dataclasses.dataclass
class StreamConfig:
    """One named data stream, bound to the domain it provides.

    Parameters:
        domain: Name of the component-pool domain this stream serves. Several
            streams may serve one domain (ERA5 and IFS both feeding
            ``atmos_1deg``); they must then describe compatible datasets, which
            is checked when the loader is built.
        dataset: The underlying ace dataset configuration.
        name: The handle objectives use to reference this stream. Defaults to
            ``domain``, which is what you want when one stream serves the
            domain; name it explicitly when the source rather than the domain is
            the natural handle.
    """

    domain: str
    dataset: ConcatDatasetConfig | MergeDatasetConfig | XarrayDataConfig
    name: str | None = None

    def __post_init__(self):
        if not self.domain:
            raise ValueError("A stream must name a non-empty domain.")
        if self.name is not None and not self.name:
            raise ValueError(
                f"Stream for domain {self.domain!r} has an empty name; omit "
                "`name` to default it to the domain."
            )

    @property
    def stream_name(self) -> str:
        """The stream's name, defaulting to its domain.

        Distinct from the ``name`` field so the field keeps exactly what the
        user wrote (and round-trips through ``dataclasses.asdict``) while
        callers get a resolved ``str``.
        """
        return self.domain if self.name is None else self.name

    def get_dataset(
        self, names: list[str], n_timesteps: IntSchedule
    ) -> tuple[DatasetABC, DatasetProperties]:
        return self.dataset.build(names, n_timesteps)


@dataclasses.dataclass
class TranslateDataLoaderConfig:
    """Configuration for a translate training or validation data loader.

    Parameters:
        streams: The data streams to load, in any order. Stream names (see
            :attr:`StreamConfig.stream_name`) must be unique.
        batch_size: Number of samples per batch, per stream. Must be divisible
            by the number of data-parallel ranks.
        num_data_workers: Number of parallel workers to use for data loading.
        prefetch_factor: How many batches a single data worker will attempt to
            hold in host memory at a given time.

    Note:
        ace's ``time_buffer`` window-reuse options are not offered here. They
        draw several output batches from one preloaded window, which would break
        the index pairing that keeps a group's streams on a shared time index.
    """

    streams: list[StreamConfig]
    batch_size: int
    num_data_workers: int = 0
    prefetch_factor: int | None = None

    def __post_init__(self):
        if not self.streams:
            raise ValueError("At least one data stream must be configured.")
        names = [stream.stream_name for stream in self.streams]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "Stream names must be unique (they are the handles objectives "
                f"reference); found duplicates: {duplicates}. Set an explicit "
                "`name` on streams sharing a domain."
            )
        dist = Distributed.get_instance()
        if self.batch_size % dist.total_data_parallel_ranks != 0:
            raise ValueError(
                "batch_size must be divisible by the number of data-parallel "
                f"workers, got {self.batch_size} and "
                f"{dist.total_data_parallel_ranks}"
            )
        self._zarr_engine_used = any(
            stream.dataset.zarr_engine_used for stream in self.streams
        )

    @property
    def stream_configs(self) -> dict[str, StreamConfig]:
        """The streams keyed by name."""
        return {stream.stream_name: stream for stream in self.streams}

    @property
    def zarr_engine_used(self) -> bool:
        """Whether any configured stream reads through the Zarr engine."""
        return self._zarr_engine_used
