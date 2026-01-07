"""Contains code relating to loading (fine, coarse) examples for downscaling."""

import dataclasses
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Literal, Self, cast

import torch
import torch.utils.data
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.dataset import DatasetItem
from fme.core.dataset.properties import DatasetProperties
from fme.core.device import get_device, move_tensordict_to_device
from fme.core.generics.data import SizedMap
from fme.core.typing_ import TensorMapping
from fme.downscaling.data.patching import Patch, get_patches
from fme.downscaling.data.topography import Topography
from fme.downscaling.data.utils import (
    BatchedLatLonCoordinates,
    ClosedInterval,
    check_leading_dim,
    expand_and_fold_tensor,
    get_offset,
    null_generator,
    paired_shuffle,
    scale_tuple,
)


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

    def __post_init__(self):
        self._validate()
        self._horizontal_shape = next(iter(self.data.values())).shape[-2:]

    def __iter__(self):
        return iter([self.data, self.time, self.latlon_coordinates])

    @property
    def horizontal_shape(self) -> tuple[int, int]:
        return self._horizontal_shape

    def to_device(self) -> "BatchItem":
        device_latlon = LatLonCoordinates(
            lat=self.latlon_coordinates.lat.to(get_device()),
            lon=self.latlon_coordinates.lon.to(get_device()),
        )
        return BatchItem(
            move_tensordict_to_device(self.data),
            self.time,
            device_latlon,
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
        return True


# TODO: If we move the subsetting, we still have to handle the latlon coordinates
class HorizontalSubsetDataset(torch.utils.data.Dataset):
    """Subsets the horizontal latitude-longitude dimensions of a dataset."""

    def __init__(
        self,
        dataset: XarrayConcat,
        properties: DatasetProperties,
        lat_interval: ClosedInterval,
        lon_interval: ClosedInterval,
    ):
        self.dataset = dataset
        self._properties = properties
        self.lat_interval = lat_interval
        self.lon_interval = lon_interval

        if not isinstance(properties.horizontal_coordinates, LatLonCoordinates):
            raise NotImplementedError(
                "Horizontal coordinates must be of type LatLonCoordinates"
            )

        self._orig_coords: LatLonCoordinates = properties.horizontal_coordinates
        lats = torch.tensor(
            [
                i
                for i in range(len(self._orig_coords.lat))
                if float(self._orig_coords.lat[i]) in self.lat_interval
            ]
        )
        lons = torch.tensor(
            [
                i
                for i in range(len(self._orig_coords.lon))
                if float(self._orig_coords.lon[i]) in self.lon_interval
            ]
        )

        if (self.lon_interval.stop != float("inf")) and (
            torch.any(self._orig_coords.lon < self.lon_interval.stop - 360.0)
        ):
            lon_max = self._orig_coords.lon.max()
            raise NotImplementedError(
                f"lon wraparound not implemented, received lon_max {lon_max} but "
                f"expected lon_max > {self.lon_interval.stop - 360.0}"
            )
        if (self.lon_interval.start != -float("inf")) and (
            torch.any(self._orig_coords.lon > self.lon_interval.start + 360.0)
        ):
            lon_min = self._orig_coords.lon.min()
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
            lat=self._orig_coords.lat[self.mask_indices.lat],
            lon=self._orig_coords.lon[self.mask_indices.lon],
        )
        self._area_weights = self._latlon_coordinates.area_weights

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key) -> DatasetItem:
        batch, times, _, epoch = self.dataset[key]
        batch = {
            k: v[
                ...,
                self.mask_indices.lat.unsqueeze(1),
                self.mask_indices.lon.unsqueeze(0),
            ]
            for k, v in batch.items()
        }
        return batch, times, self._properties.all_labels, epoch


class BatchItemDatasetAdapter(torch.utils.data.Dataset):
    """
    Adjusts output of dataset to return a BatchItem.
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

    def __getitem__(self, idx) -> BatchItem:
        fields, time, _, epoch = self._dataset[idx]
        fields = {k: v.squeeze() for k, v in fields.items()}
        field_example = next(iter(fields.values()))

        # This is hardcoded by the model DataRequirements to be
        # timestep of 1, which gets squeezed, so not expecting
        # this error to be raised
        if field_example.dim() > 2:
            raise ValueError(
                f"Expected 2D spatial data, got shape {field_example.shape}"
            )

        return BatchItem(fields, time.squeeze(), self._coordinates)

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


@dataclasses.dataclass
class GriddedData:
    _loader: torch.utils.data.DataLoader
    shape: tuple[int, int]
    dims: list[str]
    variable_metadata: Mapping[str, VariableMetadata]
    all_times: xr.CFTimeIndex
    topography: Topography | None

    @property
    def loader(self) -> DataLoader[BatchItem]:
        def on_device(batch: BatchItem) -> BatchItem:
            return batch.to_device()

        return SizedMap(on_device, self._loader)

    @property
    def topography_downscale_factor(self) -> int | None:
        if self.topography:
            if (
                self.topography.shape[0] % self.shape[0] != 0
                or self.topography.shape[1] % self.shape[1] != 0
            ):
                raise ValueError(
                    "Topography shape must be evenly divisible by data shape. "
                    f"Got topography {self.topography.shape} and data {self.shape}"
                )
            return self.topography.shape[0] // self.shape[0]
        else:
            return None

    def get_generator(
        self,
    ) -> Iterator[tuple["BatchData", Topography | None]]:
        for batch in self.loader:
            yield (batch, self.topography)

    def get_patched_generator(
        self,
        yx_patch_extent: tuple[int, int],
        overlap: int = 0,
        drop_partial_patches: bool = True,
        random_offset: bool = False,
    ) -> Iterator[tuple["BatchData", Topography | None]]:
        patched_generator = patched_batch_gen_from_loader(
            loader=self.loader,
            topography=self.topography,
            coarse_yx_extent=self.shape,
            coarse_yx_patch_extent=yx_patch_extent,
            downscale_factor=self.topography_downscale_factor,
            coarse_overlap=overlap,
            drop_partial_patches=drop_partial_patches,
            random_offset=random_offset,
        )

        return cast(
            Iterator[tuple[BatchData, Topography | None]],
            patched_generator,
        )


@dataclasses.dataclass
class PairedGriddedData:
    _loader: torch.utils.data.DataLoader
    coarse_shape: tuple[int, int]
    downscale_factor: int
    dims: list[str]
    variable_metadata: Mapping[str, VariableMetadata]
    all_times: xr.CFTimeIndex
    topography: Topography | None

    @property
    def loader(self) -> DataLoader[PairedBatchItem]:
        def on_device(batch: PairedBatchItem) -> PairedBatchItem:
            return batch.to_device()

        return SizedMap(on_device, self._loader)

    def get_generator(
        self,
    ) -> Iterator[tuple["PairedBatchData", Topography | None]]:
        for batch in self.loader:
            yield (batch, self.topography)

    def get_patched_generator(
        self,
        coarse_yx_patch_extent: tuple[int, int],
        overlap: int = 0,
        drop_partial_patches: bool = True,
        random_offset: bool = False,
        shuffle: bool = False,
    ) -> Iterator[tuple["PairedBatchData", Topography | None]]:
        patched_generator = patched_batch_gen_from_paired_loader(
            self.loader,
            self.topography,
            coarse_yx_extent=self.coarse_shape,
            coarse_yx_patch_extent=coarse_yx_patch_extent,
            downscale_factor=self.downscale_factor,
            coarse_overlap=overlap,
            drop_partial_patches=drop_partial_patches,
            random_offset=random_offset,
            shuffle=shuffle,
        )
        return cast(
            Iterator[tuple[PairedBatchData, Topography | None]],
            patched_generator,
        )


@dataclasses.dataclass
class BatchData:
    """
    Downscaling dataset grouping with a leading batch dimension.
    """

    data: TensorMapping
    time: xr.DataArray
    latlon_coordinates: BatchedLatLonCoordinates

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

        # TODO: temporary constraint for only 1 leading batch dimension
        if len(leading_dim) != 1:
            raise NotImplementedError("Only 1 leading batch dimension is supported")

        return leading_dim

    def __post_init__(self):
        leading_dim = self._validate()
        self._len = leading_dim[0]
        self._horizontal_shape = self[0].horizontal_shape
        self.is_patched = False

    @property
    def horizontal_shape(self) -> tuple[int, int]:
        return self._horizontal_shape

    @classmethod
    def from_sequence(
        cls,
        items: Sequence[BatchItem],
        dim_name: str = "batch",
    ) -> Self:
        data, times, latlon_coordinates = zip(*items)

        return cls(
            torch.utils.data.default_collate(data),
            xr.concat(times, dim_name),
            BatchedLatLonCoordinates.from_sequence(latlon_coordinates),
        )

    def to_device(self) -> "BatchData":
        return BatchData(
            move_tensordict_to_device(self.data),
            self.time,
            self.latlon_coordinates.to_device(),
        )

    def __getitem__(self, k):
        return BatchItem(
            {key: value[k].squeeze() for key, value in self.data.items()},
            self.time[k],
            self.latlon_coordinates[k],
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
        return BatchData(data, time, latlon_coordinates)

    def latlon_slice(
        self,
        lat_slice: slice,
        lon_slice: slice,
    ) -> "BatchData":
        # This method differs from using HorizontalSubsetDataset because the subsets
        # are specified as index slices rather than coordinate ranges. This is useful
        # for dividing a region into patches.
        sliced_data = {k: v[..., lat_slice, lon_slice] for k, v in self.data.items()}
        sliced_latlon = BatchedLatLonCoordinates(
            lat=self.latlon_coordinates.lat[..., lat_slice],
            lon=self.latlon_coordinates.lon[..., lon_slice],
            dims=self.latlon_coordinates.dims,
        )
        return BatchData(
            data=sliced_data,
            time=self.time,
            latlon_coordinates=sliced_latlon,
        )

    def apply_patch(
        self, patch: Patch, type: Literal["input", "output"]
    ) -> "BatchData":
        if self.is_patched:
            raise ValueError("Patching previously patched data is not supported.")

        use_slice = patch.input_slice if type == "input" else patch.output_slice

        data = self.latlon_slice(lat_slice=use_slice.y, lon_slice=use_slice.x)
        data.is_patched = True
        return data

    def generate_from_patches(
        self, patches: list[Patch], patch_type: Literal["input", "output"] = "input"
    ) -> Iterator["BatchData"]:
        for patch in patches:
            yield self.apply_patch(patch, patch_type)


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

    def generate_from_patches(
        self,
        coarse_patches: list[Patch],
        fine_patches: list[Patch],
    ) -> Iterator["PairedBatchData"]:
        coarse_gen = self.coarse.generate_from_patches(coarse_patches)
        fine_gen = self.fine.generate_from_patches(fine_patches)

        for coarse_batch, fine_batch in zip(coarse_gen, fine_gen):
            yield PairedBatchData(fine=fine_batch, coarse=coarse_batch)


class ContiguousDistributedSampler(DistributedSampler):
    """Distributes contiguous chunks of data across ranks.
    This is useful when we desire generated chunks to be contiguous
    in time, for example generating new datasets for downstream training.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, drop_last: bool = False):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            drop_last=drop_last,
        )

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Deterministically split data into contiguous chunks
        indices = list(range(len(self.dataset)))

        if self.drop_last:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]

        # Subsample contiguous chunk for this rank
        total_size = len(indices)
        chunk_size = total_size // self.num_replicas
        start = self.rank * chunk_size
        end = start + chunk_size if self.rank != self.num_replicas - 1 else total_size

        return iter(indices[start:end])


def _get_paired_patches(
    coarse_yx_extent: tuple[int, int],
    coarse_yx_patch_extent: tuple[int, int],
    coarse_overlap: int,
    downscale_factor: int | None,
    random_offset: bool = False,
    shuffle: bool = False,
    drop_partial_patches: bool = True,
) -> tuple[list[Patch], list[Patch] | None]:
    coarse_y_offset = get_offset(
        random_offset, coarse_yx_extent[0], coarse_yx_patch_extent[0]
    )
    coarse_x_offset = get_offset(
        random_offset, coarse_yx_extent[1], coarse_yx_patch_extent[1]
    )
    coarse_patches = get_patches(
        yx_extent=coarse_yx_extent,
        yx_patch_extent=coarse_yx_patch_extent,
        overlap=coarse_overlap,
        drop_partial_patches=drop_partial_patches,
        y_offset=coarse_y_offset,
        x_offset=coarse_x_offset,
    )
    if downscale_factor is not None:
        fine_yx_extent = scale_tuple(coarse_yx_extent, downscale_factor)
        fine_yx_patch_extent = scale_tuple(coarse_yx_patch_extent, downscale_factor)
        fine_patches = get_patches(
            yx_extent=fine_yx_extent,
            yx_patch_extent=fine_yx_patch_extent,
            overlap=coarse_overlap * downscale_factor,
            drop_partial_patches=drop_partial_patches,
            y_offset=coarse_y_offset * downscale_factor,
            x_offset=coarse_x_offset * downscale_factor,
        )
        if shuffle:
            # Shuffling is only relevant for training, which is over paired data
            coarse_patches, fine_patches = paired_shuffle(coarse_patches, fine_patches)
    else:
        fine_patches = None
    return coarse_patches, fine_patches


def patched_batch_gen_from_loader(
    loader: DataLoader[BatchItem],
    topography: Topography | None,
    coarse_yx_extent: tuple[int, int],
    coarse_yx_patch_extent: tuple[int, int],
    downscale_factor: int | None,
    coarse_overlap: int = 0,
    drop_partial_patches: bool = True,
    random_offset: bool = False,
    shuffle: bool = False,
) -> Iterator[tuple[BatchData, Topography | None]]:
    for batch in loader:
        coarse_patches, fine_patches = _get_paired_patches(
            coarse_yx_extent=coarse_yx_extent,
            coarse_yx_patch_extent=coarse_yx_patch_extent,
            coarse_overlap=coarse_overlap,
            downscale_factor=downscale_factor,
            random_offset=random_offset,
            shuffle=shuffle,
            drop_partial_patches=drop_partial_patches,
        )
    batch_data_patches = batch.generate_from_patches(coarse_patches)

    if topography is not None:
        if fine_patches is None:
            raise ValueError(
                "Topography provided but downscale_factor is None, cannot "
                "generate fine patches."
            )
        topography_patches = topography.generate_from_patches(fine_patches)
    else:
        topography_patches = null_generator(len(coarse_patches))

    # Combine outputs from both generators
    yield from zip(batch_data_patches, topography_patches)


def patched_batch_gen_from_paired_loader(
    loader: DataLoader[PairedBatchItem],
    topography: Topography | None,
    coarse_yx_extent: tuple[int, int],
    coarse_yx_patch_extent: tuple[int, int],
    downscale_factor: int,
    coarse_overlap: int = 0,
    drop_partial_patches: bool = True,
    random_offset: bool = False,
    shuffle: bool = False,
) -> Iterator[tuple[PairedBatchData, Topography | None]]:
    for batch in loader:
        coarse_patches, fine_patches = _get_paired_patches(
            coarse_yx_extent=coarse_yx_extent,
            coarse_yx_patch_extent=coarse_yx_patch_extent,
            coarse_overlap=coarse_overlap,
            downscale_factor=downscale_factor,
            random_offset=random_offset,
            shuffle=shuffle,
            drop_partial_patches=drop_partial_patches,
        )
        batch_data_patches = batch.generate_from_patches(coarse_patches, fine_patches)

        if topography is not None:
            if fine_patches is None:
                raise ValueError(
                    "Topography provided but downscale_factor is None, cannot "
                    "generate fine patches."
                )
            topography_patches = topography.generate_from_patches(fine_patches)
        else:
            topography_patches = null_generator(len(coarse_patches))

        # Combine outputs from both generators
        yield from zip(batch_data_patches, topography_patches)
