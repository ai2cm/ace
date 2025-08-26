"""Contains code relating to loading (fine, coarse) examples for downscaling."""

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from typing import Generic, Self, TypeVar

import torch
import torch.utils.data
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.properties import DatasetProperties
from fme.core.device import get_device, move_tensordict_to_device
from fme.core.metrics import spherical_area_weights
from fme.core.typing_ import TensorMapping
from fme.downscaling.data.utils import (
    ClosedInterval,
    check_leading_dim,
    expand_and_fold_tensor,
    scale_slice,
)


@dataclasses.dataclass
class Topography:
    data: torch.Tensor
    coordinates: LatLonCoordinates


def get_normalized_topography(path: str, topography_name: str = "HGTsfc"):
    if path.endswith(".zarr"):
        topography = xr.open_zarr(path, mask_and_scale=False)[topography_name]
    else:
        topography = xr.open_dataset(path, mask_and_scale=False)[topography_name]
    if "time" in topography.dims:
        topography = topography.isel(time=0).squeeze()
    if len(topography.shape) != 2:
        raise ValueError(f"unexpected shape {topography.shape} for topography")
    topography_normalized = (topography - topography.mean()) / topography.std()
    return torch.tensor(topography_normalized.values)


def get_topography_downscale_factor(
    topography_shape: tuple[int, ...], data_coords_shape: tuple[int, ...]
):
    if len(topography_shape) != 2 or len(data_coords_shape) != 2:
        raise ValueError(
            f"Expected 2D shapes for topography {topography_shape} and "
            f"data coordinates {data_coords_shape}, got {len(topography_shape)}D "
            f"and {len(data_coords_shape)}D."
        )
    if (
        topography_shape[0] % data_coords_shape[0] != 0
        or topography_shape[1] % data_coords_shape[1] != 0
    ):
        raise ValueError(
            f"Topography shape {topography_shape} must be evenly "
            f"divisible by horizontal shape {data_coords_shape}"
        )
    topography_downscale_factor = topography_shape[0] // data_coords_shape[0]
    if topography_downscale_factor != topography_shape[1] // data_coords_shape[1]:
        raise ValueError(
            f"Topography shape {topography_shape} must have the same scale factor "
            "between lat and lon dimensions as data coordinates "
            f"shape {data_coords_shape}"
        )
    return topography_downscale_factor


@dataclasses.dataclass
class BatchedLatLonCoordinates:
    """
    Container for batched latitude and longitude coordinates.
    Expects leading batch dimensions (that are the same) for
    lat and lon coordinates.
    """

    lat: torch.Tensor
    lon: torch.Tensor
    dims: list[str] = dataclasses.field(default_factory=lambda: ["batch", "lat", "lon"])

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.lat.dim() != 2 or self.lon.dim() != 2:
            raise ValueError(
                f"Expected 2D lat and lon coordinates, got shapes {self.lat.shape} "
                f"and {self.lon.shape}."
            )

        if self.lat.shape[0] != self.lon.shape[0]:
            raise ValueError(
                f"Latitude batch dimension {self.lat.shape[0]} does not match "
                f"longitude batch dimension {self.lon.shape[0]}"
            )

    @classmethod
    def from_sequence(
        cls,
        items: Sequence[LatLonCoordinates],
    ) -> "BatchedLatLonCoordinates":
        lats = torch.utils.data.default_collate([i.lat for i in items])
        lons = torch.utils.data.default_collate([i.lon for i in items])
        return BatchedLatLonCoordinates(lats, lons)

    @property
    def area_weights(self) -> torch.Tensor:
        return spherical_area_weights(self.lat, self.lon.shape[-1])

    def to_device(self) -> "BatchedLatLonCoordinates":
        device = get_device()
        return BatchedLatLonCoordinates(self.lat.to(device), self.lon.to(device))

    def __getitem__(self, k):
        lats = self.lat[k]
        lons = self.lon[k]

        return LatLonCoordinates(lat=lats, lon=lons)

    def __eq__(self, other):
        return torch.equal(self.lat, other.lat) and torch.equal(self.lon, other.lon)

    def __len__(self):
        return self.lat.shape[0]


@dataclasses.dataclass
class BatchItem:
    """
    Defines a single downscaling data item with assosciated metadata.  Should
    not contain any leading batch dimensions.

    Note that attached topography is usually normalized for special handling
    inside the downscaling model.
    """

    data: TensorMapping
    time: xr.DataArray
    latlon_coordinates: LatLonCoordinates
    topography: torch.Tensor | None = None

    def _validate(self):
        for key, value in self.data.items():
            if value.dim() != 2:
                raise ValueError(
                    f"Expected 2D spatial data, got shape {value.shape} ({key})"
                )
        if self.time.shape != ():
            raise ValueError(f"Expected scalar time, got shape {self.time.shape}")
        if len(self.latlon_coordinates.lat.shape) != 1:
            raise ValueError(
                "Expected 1D lat coordinates, got shape "
                f"{self.latlon_coordinates.lat.shape}"
            )
        if self.latlon_coordinates.lon.dim() != 1:
            raise ValueError(
                "Expected 1D lon coordinates, got shape "
                f"{self.latlon_coordinates.lon.shape}"
            )
        if self.topography is not None and self.topography.dim() != 2:
            raise ValueError(
                f"Expected 2D topography, got shape {self.topography.shape}"
            )

    def __post_init__(self):
        self._validate()
        self._horizontal_shape = next(iter(self.data.values())).shape[-2:]

    def __iter__(self):
        return iter([self.data, self.time, self.latlon_coordinates, self.topography])

    @property
    def horizontal_shape(self) -> tuple[int, int]:
        return self._horizontal_shape

    def to_device(self) -> "BatchItem":
        device_latlon = LatLonCoordinates(
            lat=self.latlon_coordinates.lat.to(get_device()),
            lon=self.latlon_coordinates.lon.to(get_device()),
        )
        if self.topography is not None:
            topography = self.topography.to(get_device())
        else:
            topography = None

        return BatchItem(
            move_tensordict_to_device(self.data),
            self.time,
            device_latlon,
            topography,
        )

    def __eq__(self, value) -> bool:
        for key in self.data.keys():
            if key not in value.data.keys():
                return False
            if not torch.equal(self.data[key], value.data[key]):
                return False
        if not self.time == value.time:
            return False
        if not self.latlon_coordinates == value.latlon_coordinates:
            return False
        if self.topography is not None:
            if not torch.equal(self.topography, value.topography):
                return False
        return True


# TODO: If we move the subsetting, we still have to handle the topography
#       and the latlon coordinates
class HorizontalSubsetDataset(torch.utils.data.Dataset):
    """Subsets the horizontal latitude-longitude dimensions of a dataset."""

    def __init__(
        self,
        dataset: XarrayConcat,
        properties: DatasetProperties,
        lat_interval: ClosedInterval,
        lon_interval: ClosedInterval,
        topography: torch.Tensor | None = None,
    ):
        self.dataset = dataset
        self._properties = properties
        self.lat_interval = lat_interval
        self.lon_interval = lon_interval

        if not isinstance(properties.horizontal_coordinates, LatLonCoordinates):
            raise NotImplementedError(
                "Horizontal coordinates must be of type LatLonCoordinates"
            )

        coords: LatLonCoordinates = properties.horizontal_coordinates
        lats = torch.tensor(
            [
                i
                for i in range(len(coords.lat))
                if float(coords.lat[i]) in self.lat_interval
            ]
        )
        lons = torch.tensor(
            [
                i
                for i in range(len(coords.lon))
                if float(coords.lon[i]) in self.lon_interval
            ]
        )

        if (self.lon_interval.stop != float("inf")) and (
            torch.any(coords.lon < self.lon_interval.stop - 360.0)
        ):
            lon_max = coords.lon.max()
            raise NotImplementedError(
                f"lon wraparound not implemented, received lon_max {lon_max} but "
                f"expected lon_max > {self.lon_interval.stop - 360.0}"
            )
        if (self.lon_interval.start != -float("inf")) and (
            torch.any(coords.lon > self.lon_interval.start + 360.0)
        ):
            lon_min = coords.lon.min()
            raise NotImplementedError(
                f"lon wraparound not implemented, received lon_min {lon_min} but "
                f"expected lon_min < {self.lon_interval.start + 360.0}"
            )

        assert lats.numel() > 0, "No latitudes found in the specified range."
        assert lons.numel() > 0, "No longitudes found in the specified range."

        self.mask_indices = LatLonCoordinates(
            lat=lats,
            lon=lons,
        )
        self._latlon_coordinates = LatLonCoordinates(
            lat=coords.lat[self.mask_indices.lat],
            lon=coords.lon[self.mask_indices.lon],
        )
        self._area_weights = self._latlon_coordinates.area_weights
        self._full_topography = topography
        self._full_shape = (
            coords.lat.numel(),
            coords.lon.numel(),
        )
        if self._full_topography is not None:
            self._topography_mask = self._get_topography_mask(
                self.mask_indices,
                self._full_topography.shape,
                self._full_shape,
            )
        else:
            self._topography_mask = None

    def _get_topography_mask(
        self, data_mask_indices, topography_shape, data_coords_shape
    ):
        """
        Topography is allowed to be higher resolution than the data,
        as a common use case is to load fine topography as an input
        when loading coarse input data.
        """
        topography_downscale_factor = get_topography_downscale_factor(
            topography_shape, data_coords_shape
        )
        lat_mask = torch.arange(
            (data_mask_indices.lat[0]) * topography_downscale_factor,
            (data_mask_indices.lat[-1] + 1) * topography_downscale_factor,
        )
        lon_mask = torch.arange(
            (data_mask_indices.lon[0]) * topography_downscale_factor,
            (data_mask_indices.lon[-1] + 1) * topography_downscale_factor,
        )
        mask = (lat_mask.unsqueeze(1), lon_mask.unsqueeze(0))

        return mask

    @property
    def variable_metadata(self) -> dict[str, VariableMetadata]:
        return self._properties.variable_metadata

    @property
    def vertical_coordinate(self):
        return self._properties.vertical_coordinate

    @property
    def is_remote(self) -> bool:
        return self._properties.is_remote

    @property
    def subset_latlon_coordinates(self) -> LatLonCoordinates:
        return self._latlon_coordinates

    @property
    def subset_topography(self) -> torch.Tensor | None:
        if self._full_topography is not None:
            return self._full_topography[*self._topography_mask]
        else:
            return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key) -> tuple[TensorMapping, xr.DataArray, set[str]]:
        batch, times, _ = self.dataset[key]
        batch = {
            k: v[
                ...,
                self.mask_indices.lat.unsqueeze(1),
                self.mask_indices.lon.unsqueeze(0),
            ]
            for k, v in batch.items()
        }
        return batch, times, self._properties.all_labels


class BatchItemDatasetAdapter(torch.utils.data.Dataset):
    """
    Adjusts output of dataset to return a BatchItem.
    """

    def __init__(
        self,
        dataset: HorizontalSubsetDataset | XarrayConcat,
        coordinates: LatLonCoordinates,
        topography: torch.Tensor | None = None,
        properties: DatasetProperties | None = None,
    ):
        self._dataset = dataset
        self._coordinates = coordinates
        self._topography = topography
        self._properties = properties

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx) -> BatchItem:
        fields, time, _ = self._dataset[idx]
        fields = {k: v.squeeze() for k, v in fields.items()}
        field_example = next(iter(fields.values()))

        # This is hardcoded by the model DataRequirements to be
        # timestep of 1, which gets squeezed, so not expecting
        # this error to be raised
        if field_example.dim() > 2:
            raise ValueError(
                f"Expected 2D spatial data, got shape {field_example.shape}"
            )

        return BatchItem(fields, time.squeeze(), self._coordinates, self._topography)

    @property
    def variable_metadata(self) -> dict[str, VariableMetadata]:
        if self._properties is None:
            raise ValueError("Properties not set for this dataset.")
        return self._properties.variable_metadata


@dataclasses.dataclass
class PairedBatchItem:
    """
    Container for pair of fine and coarse batch item.
    Used for convenience to validate and collate items
    into a batch.
    """

    fine: BatchItem
    coarse: BatchItem

    def _validated_scale_factor(self):
        fine = self.fine.horizontal_shape
        coarse = self.coarse.horizontal_shape

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

    def _validate(self):
        if not self.fine.time == self.coarse.time:
            raise ValueError("Time must match between fine and coarse items.")
        self.downscale_factor = self._validated_scale_factor()

    def __post_init__(self):
        self._validate()

    def to_device(self) -> "PairedBatchItem":
        return PairedBatchItem(self.fine.to_device(), self.coarse.to_device())

    def __iter__(self):
        return iter([self.fine, self.coarse])


class FineCoarsePairedDataset(torch.utils.data.Dataset):
    """
    A torch dataset that will return a paired fine and coarse batch item.
    from input fine and coarse datasets.
    """

    def __init__(
        self,
        fine: BatchItemDatasetAdapter,
        coarse: BatchItemDatasetAdapter,
    ):
        self.fine = fine
        self.coarse = coarse

        self._validate()

    def _validate(self):
        """Check that datasets are compatible."""
        if not len(self.fine) == len(self.coarse):
            raise ValueError("Datasets must have the same number of items.")

    def __len__(self):
        return len(self.fine)

    def __getitem__(self, idx) -> PairedBatchItem:
        return PairedBatchItem(self.fine[idx], self.coarse[idx])


T = TypeVar("T", covariant=True)
U = TypeVar("U")


class SizedMap(Generic[T, U], Sized, Iterable[U]):
    def __init__(self, func: Callable[[T], U], iterable: DataLoader[T]):
        self._func = func
        self._iterable = iterable

    def __len__(self) -> int:
        return len(self._iterable)

    def __iter__(self) -> Iterator[U]:
        return map(self._func, self._iterable)


@dataclasses.dataclass
class GriddedData:
    _loader: torch.utils.data.DataLoader
    shape: tuple[int, int]
    dims: list[str]
    variable_metadata: Mapping[str, VariableMetadata]
    all_times: xr.CFTimeIndex

    @property
    def loader(self) -> DataLoader[BatchItem]:
        def on_device(batch: BatchItem) -> BatchItem:
            return batch.to_device()

        return SizedMap(on_device, self._loader)


@dataclasses.dataclass
class PairedGriddedData:
    _loader: torch.utils.data.DataLoader
    coarse_shape: tuple[int, int]
    downscale_factor: int
    dims: list[str]
    variable_metadata: Mapping[str, VariableMetadata]
    all_times: xr.CFTimeIndex

    @property
    def loader(self) -> DataLoader[PairedBatchItem]:
        def on_device(batch: PairedBatchItem) -> PairedBatchItem:
            return batch.to_device()

        return SizedMap(on_device, self._loader)


@dataclasses.dataclass
class BatchData:
    """
    Downscaling dataset grouping with a leading batch dimension.

    Note that attached topography is usually normalized for special handling
    inside the downscaling model.
    """

    data: TensorMapping
    time: xr.DataArray
    latlon_coordinates: BatchedLatLonCoordinates
    topography: torch.Tensor | None = None

    def _validate(self):
        leading_dim = None
        for key, value in self.data.items():
            if leading_dim is None:
                leading_dim = value.shape[:-2]
            else:
                check_leading_dim(f"data {key}", value.shape[:-2], leading_dim)
        if leading_dim is None:
            raise ValueError("Data must have at least one variable")

        check_leading_dim("time", self.time.shape, leading_dim)
        check_leading_dim("lat", self.latlon_coordinates.lat.shape[:-1], leading_dim)
        check_leading_dim("lon", self.latlon_coordinates.lon.shape[:-1], leading_dim)
        if self.topography is not None:
            check_leading_dim("topography", self.topography.shape[:-2], leading_dim)

        # TODO: temporary constraint for only 1 leading batch dimension
        if len(leading_dim) != 1:
            raise NotImplementedError("Only 1 leading batch dimension is supported")

        return leading_dim

    def __post_init__(self):
        leading_dim = self._validate()
        self._len = leading_dim[0]
        self._horizontal_shape = self[0].horizontal_shape
        if self.topography is not None:
            self._topography_downscale_factor = get_topography_downscale_factor(
                self.topography.shape[-2:], self._horizontal_shape
            )
        else:
            self._topography_downscale_factor = None

    @property
    def horizontal_shape(self) -> tuple[int, int]:
        return self._horizontal_shape

    @classmethod
    def from_sequence(
        cls,
        items: Sequence[BatchItem],
        dim_name: str = "batch",
    ) -> Self:
        data, times, latlon_coordinates, fine_topographies = zip(*items)

        if any(topo is None for topo in fine_topographies):
            fine_topography = None
        else:
            fine_topography = torch.utils.data.default_collate(fine_topographies)

        return cls(
            torch.utils.data.default_collate(data),
            xr.concat(times, dim_name),
            BatchedLatLonCoordinates.from_sequence(latlon_coordinates),
            fine_topography,
        )

    def to_device(self) -> "BatchData":
        if self.topography is not None:
            topography = self.topography.to(get_device())
        else:
            topography = None

        return BatchData(
            move_tensordict_to_device(self.data),
            self.time,
            self.latlon_coordinates.to_device(),
            topography,
        )

    def __getitem__(self, k):
        return BatchItem(
            {key: value[k].squeeze() for key, value in self.data.items()},
            self.time[k],
            self.latlon_coordinates[k],
            self.topography[k] if self.topography is not None else None,
        )

    def __len__(self):
        return self._len

    def expand_and_fold(self, num_samples, sample_dim, dim_name="sample"):
        """
        Adds a sample dimension to batch data and expands it to the number
        of samples before folding it back into a single leading dimension.
        This is useful for working with aggregregators that expect only 3D
        inputs, but we have generated an ensemble of predictions from a model.
        """
        data = {
            key: expand_and_fold_tensor(value, num_samples, sample_dim)
            for key, value in self.data.items()
        }
        if dim_name in self.time.dims:
            raise ValueError(
                f"Cannot expand time dimension {dim_name} because dim alredy exists"
            )
        time = self.time.expand_dims(dim={dim_name: num_samples}, axis=sample_dim)
        time = time.stack({"repeated_batch": time.dims})
        latlon_coordinates = BatchedLatLonCoordinates(
            lat=expand_and_fold_tensor(
                self.latlon_coordinates.lat, num_samples, sample_dim
            ),
            lon=expand_and_fold_tensor(
                self.latlon_coordinates.lon, num_samples, sample_dim
            ),
        )
        if self.topography is not None:
            topography = expand_and_fold_tensor(
                self.topography, num_samples, sample_dim
            )
        else:
            topography = None

        return BatchData(data, time, latlon_coordinates, topography)

    def latlon_slice(
        self,
        lat_slice: slice,
        lon_slice: slice,
    ) -> "BatchData":
        sliced_data = {k: v[..., lat_slice, lon_slice] for k, v in self.data.items()}
        sliced_latlon = BatchedLatLonCoordinates(
            lat=self.latlon_coordinates.lat[..., lat_slice],
            lon=self.latlon_coordinates.lon[..., lon_slice],
            dims=self.latlon_coordinates.dims,
        )
        if self.topography is not None:
            topo_lat_slice = scale_slice(lat_slice, self._topography_downscale_factor)
            topo_lon_slice = scale_slice(lon_slice, self._topography_downscale_factor)
            sliced_topo = self.topography[..., topo_lat_slice, topo_lon_slice]
        else:
            sliced_topo = None
        return BatchData(
            data=sliced_data,
            time=self.time,
            latlon_coordinates=sliced_latlon,
            topography=sliced_topo,
        )


@dataclasses.dataclass
class PairedBatchData:
    fine: BatchData
    coarse: BatchData

    def _validated_scale_factor(self):
        fine = self.fine.horizontal_shape
        coarse = self.coarse.horizontal_shape

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

    def _validate(self):
        if not len(self.fine) == len(self.coarse):
            raise ValueError("Batch must have the same number of items.")
        self.downscale_factor = self._validated_scale_factor()

    def __post_init__(self):
        self._validate()

    def to_device(self) -> "PairedBatchData":
        return PairedBatchData(self.fine.to_device(), self.coarse.to_device())

    def __getitem__(self, k):
        return PairedBatchItem(self.fine[k], self.coarse[k])

    def __len__(self):
        return len(self.fine)

    def latlon_slice(
        self,
        coarse_lat_slice: slice,
        coarse_lon_slice: slice,
        fine_lat_slice: slice,
        fine_lon_slice: slice,
    ):
        return PairedBatchData(
            fine=self.fine.latlon_slice(fine_lat_slice, fine_lon_slice),
            coarse=self.coarse.latlon_slice(coarse_lat_slice, coarse_lon_slice),
        )

    @classmethod
    def from_sequence(
        cls,
        items: Sequence[PairedBatchItem],
    ) -> Self:
        fine, coarse = zip(*items)
        return cls(BatchData.from_sequence(fine), BatchData.from_sequence(coarse))

    def expand_and_fold(self, num_samples, sample_dim, dim_name="sample"):
        fine = self.fine.expand_and_fold(num_samples, sample_dim, dim_name)
        coarse = self.coarse.expand_and_fold(num_samples, sample_dim, dim_name)
        return PairedBatchData(fine, coarse)


class ContiguousDistributedSampler(DistributedSampler):
    """Distributes contiguous chunks of data across ranks.
    This is useful when we desire generated chunks to be contiguous
    in time, for example generating new datasets for downstream training.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)

    def __iter__(self):
        # Deterministically split data into contiguous chunks
        indices = list(range(len(self.dataset)))

        # Subsample contiguous chunk for this rank
        total_size = len(indices)
        chunk_size = total_size // self.num_replicas
        start = self.rank * chunk_size
        end = start + chunk_size if self.rank != self.num_replicas - 1 else total_size

        return iter(indices[start:end])


def _subset_horizontal(
    item: BatchItem,
    slice_lat: slice,
    slice_lon: slice,
) -> BatchItem:
    dataset = {k: v[..., slice_lat, slice_lon] for k, v in item.data.items()}
    latlon_coords = LatLonCoordinates(
        lat=item.latlon_coordinates.lat[slice_lat],
        lon=item.latlon_coordinates.lon[slice_lon],
    )
    topography = item.topography
    if topography is not None:
        topography = topography[..., slice_lat, slice_lon]

    return BatchItem(
        dataset,
        item.time,
        latlon_coords,
        topography,
    )
