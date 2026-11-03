"""Video (temporal) data containers: clips of ``T`` frames with an explicit
leading time axis and per-frame physical-time features, plus the paired
fine/coarse machinery used for training and inference.
"""

import dataclasses
import datetime
from collections.abc import Iterator, Mapping, Sequence
from typing import Self

import numpy as np
import torch
import torch.utils.data
import xarray as xr
from torch.utils.data import DataLoader

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.properties import DatasetProperties
from fme.core.device import get_device, move_tensordict_to_device
from fme.core.generics.data import SizedMap
from fme.core.typing_ import TensorMapping
from fme.downscaling.data.datasets import HorizontalSubsetDataset
from fme.downscaling.data.utils import BatchedLatLonCoordinates


def compute_calendar_features(
    time: xr.DataArray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-frame UTC ``(day_of_year, second_of_day)`` for a clip.

    Returns two ``float32`` tensors of shape ``(T,)``: 1-based day-of-year and
    second-of-day in ``[0, 86400)``.
    """
    try:
        day_of_year = np.asarray(time.dt.dayofyear).reshape(-1)
        second_of_day = (
            np.asarray(time.dt.hour) * 3600
            + np.asarray(time.dt.minute) * 60
            + np.asarray(time.dt.second)
        ).reshape(-1)
    except (AttributeError, TypeError):
        values = np.asarray(time.values).reshape(-1)
        day_of_year = np.array([t.dayofyr for t in values])
        second_of_day = np.array(
            [t.hour * 3600 + t.minute * 60 + t.second for t in values]
        )

    return (
        torch.as_tensor(day_of_year, dtype=torch.float32),
        torch.as_tensor(second_of_day, dtype=torch.float32),
    )


@dataclasses.dataclass
class VideoBatchItem:
    """A single downscaling video clip (each field ``(T, lat, lon)``)."""

    data: TensorMapping
    time: xr.DataArray
    latlon_coordinates: LatLonCoordinates
    day_of_year: torch.Tensor
    second_of_day: torch.Tensor

    def _validate(self):
        n_times: int | None = None
        for key, value in self.data.items():
            if value.dim() != 3:
                raise ValueError(
                    f"Expected 3D (time, lat, lon) data, got shape {value.shape} "
                    f"({key})"
                )
            if n_times is None:
                n_times = value.shape[0]
            elif value.shape[0] != n_times:
                raise ValueError(
                    "All variables must have the same number of frames, got "
                    f"{value.shape[0]} and {n_times}."
                )
        if n_times is None:
            raise ValueError("VideoBatchItem must have at least one variable.")
        if self.time.shape != (n_times,):
            raise ValueError(
                f"Expected time of shape ({n_times},), got {self.time.shape}"
            )
        for name, feature in (
            ("day_of_year", self.day_of_year),
            ("second_of_day", self.second_of_day),
        ):
            if feature.shape != (n_times,):
                raise ValueError(
                    f"Expected {name} of shape ({n_times},), got "
                    f"{tuple(feature.shape)}"
                )
        if self.latlon_coordinates.lat.dim() != 1:
            raise ValueError(
                "Expected 1D lat coordinates, got shape "
                f"{self.latlon_coordinates.lat.shape}"
            )
        if self.latlon_coordinates.lon.dim() != 1:
            raise ValueError(
                "Expected 1D lon coordinates, got shape "
                f"{self.latlon_coordinates.lon.shape}"
            )
        return n_times

    def __post_init__(self):
        self._n_times = self._validate()
        self._horizontal_shape = next(iter(self.data.values())).shape[-2:]

    @property
    def n_times(self) -> int:
        return self._n_times

    @property
    def horizontal_shape(self) -> tuple[int, int]:
        return self._horizontal_shape

    def __iter__(self):
        return iter(
            [
                self.data,
                self.time,
                self.latlon_coordinates,
                self.day_of_year,
                self.second_of_day,
            ]
        )

    def to_device(self) -> "VideoBatchItem":
        device = get_device()
        device_latlon = LatLonCoordinates(
            lat=self.latlon_coordinates.lat.to(device),
            lon=self.latlon_coordinates.lon.to(device),
        )
        return VideoBatchItem(
            move_tensordict_to_device(self.data),
            self.time,
            device_latlon,
            self.day_of_year.to(device),
            self.second_of_day.to(device),
        )


class VideoBatchItemDatasetAdapter(torch.utils.data.Dataset):
    """Adapts a multi-timestep dataset to return a ``VideoBatchItem``,
    preserving the time axis and attaching per-frame physical-time features.
    """

    def __init__(
        self,
        dataset: HorizontalSubsetDataset | XarrayConcat,
        coordinates: LatLonCoordinates,
        properties: DatasetProperties | None = None,
    ):
        self._dataset = dataset
        self._coordinates = coordinates
        self._properties = properties

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx) -> VideoBatchItem:
        fields, time, _, _, missing_names = self._dataset[idx]
        assert (
            missing_names is None
        ), "Variable masking is not supported in downscaling."
        day_of_year, second_of_day = compute_calendar_features(time)
        return VideoBatchItem(
            dict(fields), time, self._coordinates, day_of_year, second_of_day
        )

    @property
    def variable_metadata(self) -> dict[str, VariableMetadata]:
        if self._properties is None:
            raise ValueError("Properties not set for this dataset.")
        return self._properties.variable_metadata


@dataclasses.dataclass
class VideoBatchData:
    """A batch of video clips with leading ``(batch, T, ...)`` dimensions."""

    data: TensorMapping
    time: xr.DataArray
    latlon_coordinates: BatchedLatLonCoordinates
    day_of_year: torch.Tensor
    second_of_day: torch.Tensor

    @classmethod
    def from_sequence(
        cls,
        items: Sequence[VideoBatchItem],
        dim_name: str = "batch",
    ) -> Self:
        data, times, latlon_coordinates, day_of_year, second_of_day = zip(*items)
        return cls(
            torch.utils.data.default_collate(data),
            xr.concat(times, dim_name),
            BatchedLatLonCoordinates.from_sequence(latlon_coordinates),
            torch.stack(day_of_year, dim=0),
            torch.stack(second_of_day, dim=0),
        )

    def to_device(self) -> "VideoBatchData":
        device = get_device()
        return VideoBatchData(
            move_tensordict_to_device(self.data),
            self.time,
            self.latlon_coordinates.to_device(),
            self.day_of_year.to(device),
            self.second_of_day.to(device),
        )

    def __len__(self):
        return self.day_of_year.shape[0]


class VideoFineCoarsePairedDataset(torch.utils.data.Dataset):
    """Returns a paired fine/coarse ``VideoBatchItem`` from paired datasets."""

    def __init__(
        self,
        fine: VideoBatchItemDatasetAdapter,
        coarse: VideoBatchItemDatasetAdapter,
    ):
        self.fine = fine
        self.coarse = coarse
        if len(self.fine) != len(self.coarse):
            raise ValueError("Datasets must have the same number of items.")

    def __len__(self):
        return len(self.fine)

    def __getitem__(self, idx) -> "PairedVideoBatchItem":
        return PairedVideoBatchItem(self.fine[idx], self.coarse[idx])


def _validated_video_scale_factor(
    fine: tuple[int, int], coarse: tuple[int, int]
) -> int:
    if fine[0] // coarse[0] != fine[1] // coarse[1]:
        raise ValueError(
            "Fine and coarse datasets must have the same scale factor "
            f"between lat and lon dimensions. Got fine {fine} and coarse {coarse}"
        )
    if fine[0] % coarse[0] != 0 or fine[1] % coarse[1] != 0:
        raise ValueError(
            "Fine and coarse horizontal dimensions must be evenly divisible."
            f" Got fine {fine} and coarse {coarse}"
        )
    return fine[0] // coarse[0]


@dataclasses.dataclass
class PairedVideoBatchItem:
    """Container pairing a fine and coarse ``VideoBatchItem``."""

    fine: VideoBatchItem
    coarse: VideoBatchItem

    def _validate(self):
        if self.fine.n_times != self.coarse.n_times:
            raise ValueError("Fine and coarse clips must have the same length.")
        if not (self.fine.time.values == self.coarse.time.values).all():
            raise ValueError("Time must match between fine and coarse items.")
        self.downscale_factor = _validated_video_scale_factor(
            self.fine.horizontal_shape, self.coarse.horizontal_shape
        )

    def __post_init__(self):
        self._validate()

    def to_device(self) -> "PairedVideoBatchItem":
        return PairedVideoBatchItem(self.fine.to_device(), self.coarse.to_device())

    def __iter__(self):
        return iter([self.fine, self.coarse])


@dataclasses.dataclass
class PairedVideoBatchData:
    """A batch of paired fine/coarse video clips."""

    fine: VideoBatchData
    coarse: VideoBatchData

    @classmethod
    def from_sequence(cls, items: Sequence[PairedVideoBatchItem]) -> Self:
        fine, coarse = zip(*items)
        return cls(
            VideoBatchData.from_sequence(fine),
            VideoBatchData.from_sequence(coarse),
        )

    def to_device(self) -> "PairedVideoBatchData":
        return PairedVideoBatchData(self.fine.to_device(), self.coarse.to_device())

    def __len__(self):
        return len(self.fine)


@dataclasses.dataclass
class PairedVideoGriddedData:
    """Loader wrapper analogous to ``PairedGriddedData`` for video clips.

    Attributes:
        coarse_shape: Horizontal shape of the coarse grid.
        downscale_factor: Fine/coarse horizontal resolution ratio.
        n_timesteps: Number of frames per clip.
        dims: Horizontal dimension names.
        variable_metadata: Per-variable units/long_name.
        clip_start_times: Each clip's start time, in loader order.
        timestep: Spacing between consecutive frames.
        frame_times: Every distinct frame time covered by a clip -- the
            sorted, deduplicated union of each clip's own frames.
        clip_start_indices: Each clip's position in ``frame_times``, i.e.
            ``frame_times[clip_start_indices] == clip_start_times``.
        fine_coords: Full-domain fine lat/lon coordinates (pre-crop).
        fine_extent_latlon_coords: Post-crop fine lat/lon coordinates.
    """

    _loader: torch.utils.data.DataLoader
    coarse_shape: tuple[int, int]
    downscale_factor: int
    n_timesteps: int
    dims: list[str]
    variable_metadata: Mapping[str, VariableMetadata]
    clip_start_times: xr.CFTimeIndex
    timestep: datetime.timedelta
    frame_times: xr.CFTimeIndex
    clip_start_indices: np.ndarray
    fine_coords: LatLonCoordinates
    fine_extent_latlon_coords: LatLonCoordinates

    @property
    def loader(self) -> DataLoader[PairedVideoBatchData]:
        def on_device(batch: PairedVideoBatchData) -> PairedVideoBatchData:
            return batch.to_device()

        return SizedMap(on_device, self._loader)

    def get_generator(self) -> Iterator["PairedVideoBatchData"]:
        yield from self.loader
