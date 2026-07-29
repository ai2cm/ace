"""A batch of named data streams, and its collate function.

:class:`TranslateBatchData` generalizes
:class:`fme.coupled.data_loading.batch_data.CoupledBatchData` from two fixed
components to an arbitrary set of named streams: where the coupled type has
``ocean_data`` and ``atmosphere_data`` fields and fans every method out over
both, this one holds a ``dict[str, BatchData]`` and fans out over its keys.

The surface is deliberately narrower than the coupled type's — per-stream
access, device movement, and the ``epoch`` a trainer needs. Time-window
manipulation (``prepend``, ``get_start``, ``remove_initial_condition``) is done
per stream by the objectives, which know which of their streams is the input,
the target, and the forcing; hoisting those onto a whole-batch fan-out would
apply them to streams that should not receive them.
"""

import dataclasses
from collections.abc import Iterator, Mapping, Sequence

from fme.ace.data_loading.batch_data import BatchData
from fme.core.dataset.dataset import DatasetItem
from fme.core.labels import LabelEncoding

__all__ = ["TranslateBatchData", "TranslateCollateFn"]


@dataclasses.dataclass
class TranslateBatchData:
    """A batch holding one :class:`BatchData` per named data stream.

    Parameters:
        streams: The batch's per-stream data, keyed by stream name.
    """

    streams: dict[str, BatchData]

    def __post_init__(self):
        if not self.streams:
            raise ValueError("A TranslateBatchData must hold at least one stream.")
        epochs = {name: batch.epoch for name, batch in self.streams.items()}
        if len(set(epochs.values())) > 1:
            raise ValueError(
                "All streams in a batch must carry the same epoch (they are "
                f"drawn in step by the trainer), got {epochs}."
            )

    @property
    def epoch(self) -> int | None:
        """The epoch every stream in this batch was drawn in.

        Consumed by ace's ``LossSchedule.init_for_epoch``, which needs the epoch
        of the data rather than of the trainer loop.
        """
        return next(iter(self.streams.values())).epoch

    def __getitem__(self, name: str) -> BatchData:
        return self.streams[name]

    def __contains__(self, name: str) -> bool:
        return name in self.streams

    def __iter__(self) -> Iterator[str]:
        return iter(self.streams)

    def __len__(self) -> int:
        return len(self.streams)

    def to_device(self) -> "TranslateBatchData":
        return TranslateBatchData(
            streams={name: batch.to_device() for name, batch in self.streams.items()}
        )

    def to_cpu(self) -> "TranslateBatchData":
        return TranslateBatchData(
            streams={name: batch.to_cpu() for name, batch in self.streams.items()}
        )

    def pin_memory(self) -> "TranslateBatchData":
        """Page-lock every stream's tensors; called by torch's DataLoader."""
        self.streams = {
            name: batch.pin_memory() for name, batch in self.streams.items()
        }
        return self

    @classmethod
    def merge(cls, batches: Sequence["TranslateBatchData"]) -> "TranslateBatchData":
        """Combine batches over disjoint stream sets into one.

        Used to assemble the independently-sampled pairing groups' batches into
        the single batch a training step sees.
        """
        streams: dict[str, BatchData] = {}
        for batch in batches:
            overlap = sorted(set(batch.streams) & set(streams))
            if overlap:
                raise ValueError(
                    f"Cannot merge batches sharing the streams {overlap}; each "
                    "stream belongs to exactly one pairing group."
                )
            streams.update(batch.streams)
        return cls(streams=streams)


class TranslateCollateFn:
    """Collates per-stream samples into a :class:`TranslateBatchData`.

    One instance serves one pairing group: its keys are that group's streams,
    and it is called with the group's paired samples (see
    :class:`fme.translate.data.dataset.PairedStreamDataset`). Defined at module
    level so it can be pickled to data-loader worker processes.
    """

    def __init__(
        self,
        horizontal_dims: Mapping[str, list[str]],
        label_encodings: Mapping[str, LabelEncoding | None],
    ):
        """
        Args:
            horizontal_dims: Each stream's horizontal dimension names, used
                when writing batches to netCDF.
            label_encodings: Each stream's label encoding, or None for a stream
                whose dataset provides no labels.
        """
        if set(horizontal_dims) != set(label_encodings):
            raise ValueError(
                "horizontal_dims and label_encodings must cover the same "
                f"streams, got {sorted(horizontal_dims)} and "
                f"{sorted(label_encodings)}."
            )
        self.horizontal_dims = dict(horizontal_dims)
        self.label_encodings = dict(label_encodings)

    def __call__(
        self, samples: Sequence[Mapping[str, DatasetItem]]
    ) -> TranslateBatchData:
        return TranslateBatchData(
            streams={
                name: BatchData.from_sample_tuples(
                    [sample[name] for sample in samples],
                    horizontal_dims=dims,
                    label_encoding=self.label_encodings[name],
                )
                for name, dims in self.horizontal_dims.items()
            }
        )
