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
from fme.core.dataset.config import XarrayDataConfig
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.getters import get_dataset
from fme.core.dataset.properties import DatasetProperties
from fme.core.device import get_device, move_tensordict_to_device, using_gpu
from fme.core.distributed import Distributed
from fme.core.metrics import spherical_area_weights
from fme.core.typing_ import TensorMapping
from fme.downscaling.requirements import DataRequirements


@dataclasses.dataclass
class XarrayEnsembleDataConfig:
    """
    Configuration for an ensemble dataset.
    This config's expand method returns a sequence of xarray datasets, each
    with the same data_config, where each individual dataset is an ensemble member
    selected from the ensemble dimension.

    Parameters:
        data_config: XarrayDataConfig for the dataset.
        ensemble_dim: Name of the ensemble dimension in the dataset.
        n_ensemble_members: Number of ensemble members to load. They will be taken
            in order from index 0 of the ensemble_dim.
    """

    data_config: XarrayDataConfig
    ensemble_dim: str
    n_ensemble_members: int

    def __post_init__(self):
        if self.n_ensemble_members <= 0:
            raise ValueError(
                f"n_ensemble_members must be > 0, got {self.n_ensemble_members}"
            )
        if self.ensemble_dim in self.data_config.isel:
            raise ValueError(
                f"Ensemble dimension {self.ensemble_dim} cannot be in the "
                "base data_config.isel"
            )

    def expand(self) -> list[XarrayDataConfig]:
        configs = []
        for i in range(self.n_ensemble_members):
            configs.append(
                dataclasses.replace(
                    self.data_config,
                    isel={self.ensemble_dim: i},
                )
            )
        return configs


def get_normalized_topography(
    configs: Sequence[XarrayDataConfig],
) -> torch.Tensor:
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


def _get_topography_downscale_factor(
    topography_shape: tuple[int, int], data_coords_shape: tuple[int, int]
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
        topography_downscale_factor = _get_topography_downscale_factor(
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
    topography: torch.Tensor | None = None

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
        if self.topography is not None:
            self._topography_downscale_factor = _get_topography_downscale_factor(
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
            topo_lat_slice = _scale_slice(lat_slice, self._topography_downscale_factor)
            topo_lon_slice = _scale_slice(lon_slice, self._topography_downscale_factor)
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


@dataclasses.dataclass
class DataLoaderConfig:
    """
    Configuration for loading downscaling data for generation.
    Input coarse dataset will be processed into batches, usually with
    a horizontal extent to define a portion of the full domain for use in
    generation.
    If the model requires topography, the dataset to use should be specified
    in the `topography` field. Topography data may be at higher resolution than
    the data, e.g. when fine topography is loaded as an input.

    Args:
        data: The dataset configuration.
        batch_size: The batch size to use for the dataloader.
        num_data_workers: The number of data workers to use for the dataloader.
            (For multi-GPU runtime, it's the number of workers per GPU.)
        strict_ensemble: Whether to enforce that the datasets to be concatened
            have the same dimensions and coordinates.
        topography: The dataset configuration for the topography data.
            If None, no topography data will be loaded.
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

    data: Sequence[XarrayDataConfig | XarrayEnsembleDataConfig]
    batch_size: int
    num_data_workers: int
    strict_ensemble: bool
    topography: XarrayDataConfig | None = None
    lat_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(-90.0, 90.0)
    )
    lon_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )
    repeat: int = 1

    @property
    def full_config(self) -> Sequence[XarrayDataConfig]:
        # Expands any XarrayEnsembleDataConfig so it is converted
        # to the equivalent sequence of XarrayDataConfig.
        all_configs = []
        for config in self.data:
            if isinstance(config, XarrayEnsembleDataConfig):
                all_configs += config.expand()
            else:
                all_configs.append(config)
        return all_configs

    @property
    def mp_context(self):
        context = None
        if self.num_data_workers == 0:
            return None
        for config in self.full_config:
            if config.engine == "zarr":
                context = "forkserver"
        return context

    def _repeat_if_requested(self, dataset: XarrayConcat) -> XarrayConcat:
        return XarrayConcat([dataset] * self.repeat)

    def get_xarray_dataset(
        self,
        names: list[str],
        n_timesteps: int,
    ) -> tuple[XarrayConcat, DatasetProperties]:
        return get_dataset(
            self.full_config,
            names,
            n_timesteps,
            strict=self.strict_ensemble,
        )

    def build_batchitem_dataset(
        self,
        dataset: XarrayConcat,
        properties: DatasetProperties,
        requires_topography: bool,
    ) -> BatchItemDatasetAdapter:
        # n_timesteps is hardcoded to 1 for downscaling, so the sample_start_times
        # are the full time range for the dataset
        if dataset.sample_n_times != 1:
            raise ValueError(
                "Downscaling data loading should always have n_timesteps=1 "
                "in model data requirements."
                f" Got {dataset.sample_n_times} instead."
            )
        dataset = self._repeat_if_requested(dataset)

        if requires_topography:
            if self.topography is None:
                raise ValueError(
                    "Topography is required for this model, but no topography "
                    "dataset was specified in the configuration."
                )
            else:
                topography = get_normalized_topography([self.topography])
        else:
            topography = None

        dataset_subset = HorizontalSubsetDataset(
            dataset,
            properties=properties,
            lat_interval=self.lat_extent,
            lon_interval=self.lon_extent,
            topography=topography,
        )
        return BatchItemDatasetAdapter(
            dataset_subset,
            dataset_subset.subset_latlon_coordinates,
            properties=properties,
            topography=dataset_subset.subset_topography,
        )

    def build(
        self,
        requirements: DataRequirements,
        dist: Distributed | None = None,
    ) -> GriddedData:
        xr_dataset, properties = self.get_xarray_dataset(
            names=requirements.coarse_names, n_timesteps=1
        )
        dataset = self.build_batchitem_dataset(
            dataset=xr_dataset,
            properties=properties,
            requires_topography=requirements.use_fine_topography,
        )
        all_times = xr_dataset.sample_start_times
        if dist is None:
            dist = Distributed.get_instance()
        # Shuffle is not used for generation, it is set to False.
        sampler = (
            ContiguousDistributedSampler(dataset) if dist.is_distributed() else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=dist.local_batch_size(int(self.batch_size)),
            num_workers=self.num_data_workers,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            collate_fn=BatchData.from_sequence,
            pin_memory=using_gpu(),
            multiprocessing_context=self.mp_context,
            persistent_workers=True if self.num_data_workers > 0 else False,
        )
        example = dataset[0]
        return GriddedData(
            dataloader,
            shape=example.horizontal_shape,
            dims=example.latlon_coordinates.dims,
            variable_metadata=dataset.variable_metadata,
            all_times=all_times,
        )


@dataclasses.dataclass
class PairedDataLoaderConfig:
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
        coarse: The coarse dataset configuration. XarrayEnsembleDataConfig
            is supported to load multiple ensemble members.
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
    coarse: Sequence[XarrayDataConfig | XarrayEnsembleDataConfig]
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

    def _mp_context(self):
        mp_context = None
        if self.num_data_workers == 0:
            return None
        for config in self.fine:
            if config.engine == "zarr":
                mp_context = "forkserver"
        for config in self.coarse_full_config:
            if config.engine == "zarr":
                mp_context = "forkserver"
        return mp_context

    @property
    def coarse_full_config(self) -> Sequence[XarrayDataConfig]:
        # Expands the coarse dataset configs so that any XarrayEnsembleDataConfig
        # is converted to the equivalent sequence of XarrayDataConfig.
        coarse_configs = []
        for config in self.coarse:
            if isinstance(config, XarrayEnsembleDataConfig):
                coarse_configs += config.expand()
            else:
                coarse_configs.append(config)
        return coarse_configs

    def build(
        self,
        train: bool,
        requirements: DataRequirements,
        dist: Distributed | None = None,
    ) -> PairedGriddedData:
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
            self.coarse_full_config,
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

        sampler: DistributedSampler | None
        if dist.is_distributed():
            if train:
                sampler = DistributedSampler(dataset, shuffle=train)
            else:
                sampler = ContiguousDistributedSampler(dataset)
        else:
            sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=dist.local_batch_size(int(self.batch_size)),
            num_workers=self.num_data_workers,
            shuffle=(sampler is None) and train,
            sampler=sampler,
            drop_last=True,
            pin_memory=using_gpu(),
            collate_fn=PairedBatchData.from_sequence,
            multiprocessing_context=self._mp_context(),
            persistent_workers=True if self.num_data_workers > 0 else False,
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

        return PairedGriddedData(
            dataloader,
            example.coarse.horizontal_shape,
            example.downscale_factor,
            example.fine.latlon_coordinates.dims,
            variable_metadata,
            all_times=all_times,
        )
