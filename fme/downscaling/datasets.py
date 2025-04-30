"""Contains code relating to loading (fine, coarse) examples for downscaling."""

import dataclasses
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Self,
    Sequence,
    Sized,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.utils.data
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.config import XarrayDataConfig
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.getters import get_dataset
from fme.core.dataset.properties import DatasetProperties
from fme.core.device import get_device, move_tensordict_to_device, using_gpu
from fme.core.distributed import Distributed
from fme.core.metrics import spherical_area_weights
from fme.core.typing_ import TensorMapping
from fme.downscaling.requirements import DataRequirements


def get_normalized_topography(configs: Sequence[XarrayDataConfig]) -> torch.Tensor:
    """
    Load the topography data from the specified path and return the normalized
    height of the topography values.

    Args:
        configs: Sequence of dataset configs corresponding to the desired
            topography data.

    Returns:
        The normalized height of the topography of shape (latitude, longitude).
    """
    topography_name = "HGTsfc"
    dataset, _ = get_dataset(configs, [topography_name], n_timesteps=1)
    example, _ = dataset[0]
    topography = example[topography_name]
    topography = topography.squeeze()
    if len(topography.shape) != 2:
        raise ValueError(f"unexpected shape {topography.shape} for topography")
    topography_normalized = (topography - topography.mean()) / topography.std()
    return topography_normalized


@dataclasses.dataclass
class ClosedInterval:
    start: float
    stop: float

    def __post_init__(self):
        assert self.start < self.stop  # Do not allow empty, start = stop

    def __contains__(self, value: float):
        return self.start <= value <= self.stop


@dataclasses.dataclass
class BatchedLatLonCoordinates:
    """
    Container for batched latitude and longitude coordinates.
    Expects leading batch dimensions (that are the same) for
    lat and lon coordinates.
    """

    lat: torch.Tensor
    lon: torch.Tensor
    dims: List[str] = dataclasses.field(default_factory=lambda: ["batch", "lat", "lon"])

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
    topography: Optional[torch.Tensor] = None

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
    def horizontal_shape(self) -> Tuple[int, int]:
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
        topography: Optional[torch.Tensor] = None,
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
                (
                    "lon wraparound not implemented, received lon_max {} but "
                    "expected lon_max > {}".format(
                        lon_max, self.lon_interval.stop - 360.0
                    )
                )
            )
        if (self.lon_interval.start != -float("inf")) and (
            torch.any(coords.lon > self.lon_interval.start + 360.0)
        ):
            lon_min = coords.lon.min()
            raise NotImplementedError(
                (
                    "lon wraparound not implemented, received lon_min {} but "
                    "expected lon_min < {}".format(
                        lon_min, self.lon_interval.start + 360.0
                    )
                )
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

        if topography is not None:
            shape = (
                coords.lat.numel(),
                coords.lon.numel(),
            )
            if topography.shape != shape:
                raise ValueError(
                    f"Topography shape {topography.shape} does not match "
                    f"horizontal coordinates shape {shape}"
                )
            self._topography = topography[
                self.mask_indices.lat.unsqueeze(1),
                self.mask_indices.lon.unsqueeze(0),
            ]
        else:
            self._topography = None

    @property
    def variable_metadata(self) -> Dict[str, VariableMetadata]:
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
    def subset_topography(self) -> Optional[torch.Tensor]:
        return self._topography

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        batch, times = self.dataset[key]
        batch = {
            k: v[
                ...,
                self.mask_indices.lat.unsqueeze(1),
                self.mask_indices.lon.unsqueeze(0),
            ]
            for k, v in batch.items()
        }
        return batch, times


class BatchItemDatasetAdapter(torch.utils.data.Dataset):
    """
    Adjusts output of dataset to return a BatchItem.
    """

    def __init__(
        self,
        dataset: Union[HorizontalSubsetDataset, XarrayConcat],
        coordinates: LatLonCoordinates,
        topography: Optional[torch.Tensor] = None,
        properties: Optional[DatasetProperties] = None,
    ):
        self._dataset = dataset
        self._coordinates = coordinates
        self._topography = topography
        self._properties = properties

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx) -> BatchItem:
        fields, time = self._dataset[idx]
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
    def variable_metadata(self) -> Dict[str, VariableMetadata]:
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


def _scale_slice(slice_: slice, scale: int) -> slice:
    if slice_ == slice(None):
        return slice_
    start = slice_.start * scale if slice_.start is not None else None
    stop = slice_.stop * scale if slice_.stop is not None else None
    return slice(start, stop)


def _slice_mapping(
    mapping: TensorMapping, lat_slice: slice, lon_slice: slice
) -> TensorMapping:
    return {k: v[..., lat_slice, lon_slice] for k, v in mapping.items()}


def _subset_horizontal(
    item: BatchItem,
    slice_lat: slice,
    slice_lon: slice,
) -> BatchItem:
    dataset = _slice_mapping(item.data, slice_lat, slice_lon)
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
    coarse_shape: Tuple[int, int]
    downscale_factor: int
    dims: List[str]
    variable_metadata: Mapping[str, VariableMetadata]
    all_times: xr.CFTimeIndex

    @property
    def loader(self) -> DataLoader[PairedBatchItem]:
        def on_device(batch: PairedBatchItem) -> PairedBatchItem:
            return batch.to_device()

        return SizedMap(on_device, self._loader)


def get_latlon_coords_from_properties(
    properties: DatasetProperties,
) -> LatLonCoordinates:
    if not isinstance(properties.horizontal_coordinates, LatLonCoordinates):
        raise NotImplementedError(
            "Horizontal coordinates must be of type LatLonCoordinates"
        )
    return properties.horizontal_coordinates


def _check_leading_dim(
    name: str, current_leading: Sequence[int], expected_leading: Sequence[int]
):
    if current_leading != expected_leading:
        raise ValueError(
            f"Expected leading dimension of {name} shape {expected_leading}, got "
            f"{current_leading}"
        )


def _expand_and_fold_tensor(
    tensor: torch.Tensor, num_samples: int, sample_dim: int
) -> torch.Tensor:
    static_shape = tensor.shape[sample_dim:]
    expanded_shape = [-1 for _ in tensor.shape]
    expanded_shape.insert(sample_dim, num_samples)
    expanded = tensor.unsqueeze(sample_dim).expand(*expanded_shape)
    return expanded.reshape(-1, *static_shape)


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
    topography: Optional[torch.Tensor] = None

    def _validate(self):
        leading_dim = None
        for key, value in self.data.items():
            if leading_dim is None:
                leading_dim = value.shape[:-2]
            else:
                _check_leading_dim(f"data {key}", value.shape[:-2], leading_dim)
        if leading_dim is None:
            raise ValueError("Data must have at least one variable")

        _check_leading_dim("time", self.time.shape, leading_dim)
        _check_leading_dim("lat", self.latlon_coordinates.lat.shape[:-1], leading_dim)
        _check_leading_dim("lon", self.latlon_coordinates.lon.shape[:-1], leading_dim)
        if self.topography is not None:
            _check_leading_dim("topography", self.topography.shape[:-2], leading_dim)

        # TODO: temporary constraint for only 1 leading batch dimension
        if len(leading_dim) != 1:
            raise NotImplementedError("Only 1 leading batch dimension is supported")

        return leading_dim

    def __post_init__(self):
        leading_dim = self._validate()
        self._len = leading_dim[0]
        self._horizontal_shape = self[0].horizontal_shape

    @property
    def horizontal_shape(self) -> Tuple[int, int]:
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
            key: _expand_and_fold_tensor(value, num_samples, sample_dim)
            for key, value in self.data.items()
        }
        if dim_name in self.time.dims:
            raise ValueError(
                f"Cannot expand time dimension {dim_name} because dim alredy exists"
            )
        time = self.time.expand_dims(dim={dim_name: num_samples}, axis=sample_dim)
        time = time.stack({"repeated_batch": time.dims})
        latlon_coordinates = BatchedLatLonCoordinates(
            lat=_expand_and_fold_tensor(
                self.latlon_coordinates.lat, num_samples, sample_dim
            ),
            lon=_expand_and_fold_tensor(
                self.latlon_coordinates.lon, num_samples, sample_dim
            ),
        )
        if self.topography is not None:
            topography = _expand_and_fold_tensor(
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
            sliced_topo = self.topography[..., lat_slice, lon_slice]
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


@dataclasses.dataclass
class DataLoaderConfig:
    """
    Configuration for loading downscaling datasets.  The input fine and
    coarse Xarray datasets will be processed into batches, usually with
    a horizontal extent to define a portion of the full domain for use in
    training or validation. Additionally, a user may specify to take
    random subsets of the initial domain by using the coarse random extent
    arguments.

    The build ensures the compatibility of the fine/coarse datasets by
    checking that the fine coordinates are evenly divisible by the coarse
    coordinates, and that the scale factors are equal.

    Args:
        fine: The fine dataset configuration.
        coarse: The coarse dataset configuration.
        batch_size: The batch size to use for the dataloader.
        num_data_workers: The number of data workers to use for the dataloader.
            (For multi-GPU runtime, it's the number of workers per GPU.)
        strict_ensemble: Whether to enforce that the datasets to be concatened
            have the same dimensions and coordinates.
        lat_extent: The latitude extent to use for the dataset specified in
            degrees (-90, 90).  The extent is inclusive, so the start and
            stop values are included in the extent.
        lon_extent: The longitude extent to use for the dataset specified in
            degrees (0, 360). The extent is inclusive, so the start and
            stop values are included in the extent.
        repeat: The number of times to repeat the underlying xarray dataset
            time dimension.  Useful to include longer sequences of small
            data for testing.
    """

    fine: Sequence[XarrayDataConfig]
    coarse: Sequence[XarrayDataConfig]
    batch_size: int
    num_data_workers: int
    strict_ensemble: bool
    lat_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(-90.0, 90.0)
    )
    lon_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )
    repeat: int = 1

    def _repeat_if_requested(self, dataset: XarrayConcat) -> XarrayConcat:
        return XarrayConcat([dataset] * self.repeat)

    def build(
        self,
        train: bool,
        requirements: DataRequirements,
        dist: Optional[Distributed] = None,
    ) -> GriddedData:
        if dist is None:
            dist = Distributed.get_instance()

        # Load initial datasets
        dataset_fine, properties_fine = get_dataset(
            self.fine,
            requirements.fine_names,
            requirements.n_timesteps,
            strict=self.strict_ensemble,
        )
        dataset_coarse, properties_coarse = get_dataset(
            self.coarse,
            requirements.coarse_names,
            requirements.n_timesteps,
            strict=self.strict_ensemble,
        )

        # n_timesteps is hardcoded to 1 for downscaling, so the sample_start_times
        # are the full time range for the dataset
        if dataset_fine.sample_n_times != 1:
            raise ValueError(
                "Downscaling data loading should always have n_timesteps=1 "
                "in model data requirements."
                f" Got {dataset_fine.sample_n_times} instead."
            )
        all_times = dataset_fine.sample_start_times

        dataset_fine = self._repeat_if_requested(dataset_fine)
        dataset_coarse = self._repeat_if_requested(dataset_coarse)

        if requirements.use_fine_topography:
            fine_topography = get_normalized_topography(self.fine)
        else:
            fine_topography = None

        # TODO: horizontal subsetting should probably live in the XarrayDatast level
        # Subset to overall horizontal domain
        dataset_fine_subset = HorizontalSubsetDataset(
            dataset_fine,
            properties=properties_fine,
            lat_interval=self.lat_extent,
            lon_interval=self.lon_extent,
            topography=fine_topography,
        )

        dataset_coarse_subset = HorizontalSubsetDataset(
            dataset_coarse,
            properties=properties_coarse,
            lat_interval=self.lat_extent,
            lon_interval=self.lon_extent,
        )

        # Convert datasets to produce BatchItems
        dataset_fine_subset = BatchItemDatasetAdapter(
            dataset_fine_subset,
            dataset_fine_subset.subset_latlon_coordinates,
            topography=dataset_fine_subset.subset_topography,
            properties=properties_fine,
        )

        dataset_coarse_subset = BatchItemDatasetAdapter(
            dataset_coarse_subset,
            dataset_coarse_subset.subset_latlon_coordinates,
            properties=properties_coarse,
        )

        dataset = FineCoarsePairedDataset(
            dataset_fine_subset,
            dataset_coarse_subset,
        )

        sampler: Optional[DistributedSampler]
        if dist.is_distributed():
            if train:
                sampler = DistributedSampler(dataset, shuffle=train)
            else:
                sampler = ContiguousDistributedSampler(dataset)
        else:
            sampler = None

        if properties_coarse.is_remote or properties_fine.is_remote:
            # GCSFS and S3FS are not fork-safe, so we need to use forkserver
            # these settings also work in the case of one or both datasets being local
            mp_context = "forkserver"
            persistent_workers = True
        else:
            mp_context = None
            persistent_workers = False

        dataloader = DataLoader(
            dataset,
            batch_size=dist.local_batch_size(int(self.batch_size)),
            num_workers=self.num_data_workers,
            shuffle=(sampler is None) and train,
            sampler=sampler,
            drop_last=True,
            pin_memory=using_gpu(),
            collate_fn=PairedBatchData.from_sequence,
            multiprocessing_context=mp_context,
            persistent_workers=persistent_workers,
        )

        example = dataset[0]
        common_metadata_keys = set(dataset_fine_subset.variable_metadata).intersection(
            dataset_coarse_subset.variable_metadata
        )
        assert all(
            dataset_fine_subset.variable_metadata[key]
            == dataset_coarse_subset.variable_metadata[key]
            for key in common_metadata_keys
        ), "Metadata for variables common to coarse and fine datasets must match."
        variable_metadata = {
            **dataset_fine_subset.variable_metadata,
            **dataset_coarse_subset.variable_metadata,
        }

        return GriddedData(
            dataloader,
            example.coarse.horizontal_shape,
            example.downscale_factor,
            example.fine.latlon_coordinates.dims,
            variable_metadata,
            all_times=all_times,
        )
