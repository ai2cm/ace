"""Contains code relating to loading (fine, coarse) examples for downscaling."""

import dataclasses
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.utils.data
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fme.core import metrics
from fme.core.data_loading._xarray import DatasetProperties
from fme.core.data_loading.config import DataLoaderConfig as CoreDataLoaderConfig
from fme.core.data_loading.config import XarrayDataConfig
from fme.core.data_loading.data_typing import (
    Dataset,
    HorizontalCoordinates,
    LatLonCoordinates,
    VariableMetadata,
)
from fme.core.data_loading.getters import get_dataset
from fme.core.data_loading.requirements import (
    DataRequirements as CoreDataRequirements,
)
from fme.core.device import using_gpu
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.typing_ import FineResCoarseResPair


def squeeze_time_dim(x: TensorMapping) -> TensorMapping:
    return {k: v.squeeze(dim=-3) for k, v in x.items()}  # (b, t=1, h, w) -> (b, h, w)


def get_topography(config: CoreDataLoaderConfig) -> torch.Tensor:
    """
    Load the topography data from the specified path and return the normalized
    height of the topography values.

    Args:
        config: Data loader configuration corresponding to the desired
            topography data.

    Returns:
        The normalized height of the topography of shape (latitude, longitude).
    """
    topography_name = "HGTsfc"
    dataset = get_dataset(
        config.dataset, CoreDataRequirements(names=[topography_name], n_timesteps=1)
    )
    example, _ = dataset[0]
    topography = example[topography_name]
    topography = topography.squeeze()
    if len(topography.shape) != 2:
        raise ValueError(f"unexpected shape {topography.shape} for topography")
    topography_normalized = (topography - topography.mean()) / topography.std()
    return topography_normalized


@dataclasses.dataclass
class BatchData:
    fine: Mapping[str, torch.Tensor]
    coarse: Mapping[str, torch.Tensor]
    times: xr.DataArray

    @classmethod
    def from_sample_tuples(
        cls,
        samples: Sequence[Tuple[TensorMapping, TensorMapping, xr.DataArray]],
        sample_dim_name: str = "sample",
    ) -> "BatchData":
        fine, coarse, times = zip(*samples)
        return cls(
            torch.utils.data.default_collate(fine),
            torch.utils.data.default_collate(coarse),
            xr.concat(times, sample_dim_name),
        )


class PairedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset1: Sequence[Tuple[TensorMapping, xr.DataArray]],
        dataset2: Sequence[Tuple[TensorMapping, xr.DataArray]],
    ):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        self._validate()

    def _validate(self):
        """Check that datasets are compatible."""
        assert len(self.dataset1) == len(
            self.dataset2
        ), "Datasets must have the same number of samples."

        batch1, times1 = self.dataset1[0]
        batch2, times2 = self.dataset2[0]
        assert all(times1 == times2), "Times must match."

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        batch1, times1 = self.dataset1[idx]
        batch2, _ = self.dataset2[idx]

        return squeeze_time_dim(batch1), squeeze_time_dim(batch2), times1.squeeze()


@dataclasses.dataclass
class ClosedInterval:
    start: float
    stop: float

    def __post_init__(self):
        assert self.start < self.stop  # Do not allow empty, start = stop

    def __contains__(self, value: float):
        return self.start <= value <= self.stop


class HorizontalSubsetDataset(Dataset):
    """Subsets the horizontal latitude-longitude dimensions of a dataset."""

    def __init__(
        self,
        dataset: Union[Dataset, torch.utils.data.ConcatDataset[Dataset]],
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
        self._horizontal_coordinates = LatLonCoordinates(
            lat=coords.lat[self.mask_indices.lat],
            lon=coords.lon[self.mask_indices.lon],
        )
        self._area_weights = metrics.spherical_area_weights(
            self._horizontal_coordinates.lat, len(self._horizontal_coordinates.lon)
        )

    @property
    def variable_metadata(self) -> Mapping[str, VariableMetadata]:
        return self._properties.variable_metadata

    @property
    def area_weights(self) -> torch.Tensor:
        return self._area_weights

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._horizontal_coordinates

    @property
    def vertical_coordinate(self):
        return self._properties.vertical_coordinate

    @property
    def is_remote(self) -> bool:
        return self._properties.is_remote

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch, times = self.dataset[idx]
        batch = {
            k: v[
                ...,
                self.mask_indices.lat.unsqueeze(1),
                self.mask_indices.lon.unsqueeze(0),
            ]
            for k, v in batch.items()
        }
        return batch, times

    def get_sample_by_time_slice(
        self, time_slice: slice
    ) -> Tuple[TensorDict, xr.DataArray]:
        sample, time = self.dataset.get_sample_by_time_slice(time_slice)  # type: ignore
        masked = {
            k: v[
                ...,
                self.mask_indices.lat.unsqueeze(1),
                self.mask_indices.lon.unsqueeze(0),
            ]
            for k, v in sample.items()
        }
        return masked, time


@dataclasses.dataclass
class GriddedData:
    loader: torch.utils.data.DataLoader
    area_weights: FineResCoarseResPair[torch.Tensor]
    horizontal_coordinates: FineResCoarseResPair[HorizontalCoordinates]
    img_shape: FineResCoarseResPair[Tuple[int, int]]
    variable_metadata: Mapping[str, VariableMetadata]
    fine_topography: Optional[torch.Tensor] = None

    def __post_init__(self):
        assert (
            self.img_shape.fine[0] % self.img_shape.coarse[0] == 0
        ), "Highres height must be divisible by lowres height"
        assert (
            self.img_shape.fine[0] // self.img_shape.coarse[0]
            == self.img_shape.fine[1] // self.img_shape.coarse[1]
        ), "Aspect ratio must match"
        self.downscale_factor: int = self.img_shape.fine[0] // self.img_shape.coarse[0]


@dataclasses.dataclass
class DataLoaderConfig:
    fine: Sequence[XarrayDataConfig]
    coarse: Sequence[XarrayDataConfig]
    batch_size: int
    num_data_workers: int
    strict_ensemble: bool
    lat_interval: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(-90.0, 90.0)
    )
    lon_interval: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )

    def build(
        self,
        train: bool,
        requirements: DataRequirements,
        dist: Optional[Distributed] = None,
    ) -> GriddedData:
        if dist is None:
            dist = Distributed.get_instance()

        dataset_fine, properties_fine = get_dataset(
            self.fine,
            CoreDataRequirements(requirements.fine_names, requirements.n_timesteps),
            strict=self.strict_ensemble,
        )
        dataset_coarse, properties_coarse = get_dataset(
            self.coarse,
            CoreDataRequirements(requirements.coarse_names, requirements.n_timesteps),
            strict=self.strict_ensemble,
        )

        dataset_fine_subset = HorizontalSubsetDataset(
            dataset_fine,
            properties=properties_fine,
            lat_interval=self.lat_interval,
            lon_interval=self.lon_interval,
        )

        dataset_coarse_subset = HorizontalSubsetDataset(
            dataset_coarse,
            properties=properties_coarse,
            lat_interval=self.lat_interval,
            lon_interval=self.lon_interval,
        )

        dataset = PairedDataset(
            dataset_fine_subset,
            dataset_coarse_subset,  # type: ignore
        )

        sampler: Optional[DistributedSampler] = (
            DistributedSampler(dataset, shuffle=train)
            if dist.is_distributed()
            else None
        )

        dataloader = DataLoader(
            dataset,
            batch_size=dist.local_batch_size(int(self.batch_size)),
            num_workers=self.num_data_workers,
            shuffle=(sampler is None) and train,
            sampler=sampler if train else None,
            drop_last=True,
            pin_memory=using_gpu(),
            collate_fn=BatchData.from_sample_tuples,
        )

        area_weights = FineResCoarseResPair(
            dataset_fine_subset.area_weights, dataset_coarse_subset.area_weights
        )
        horizontal_coordinates = FineResCoarseResPair(
            fine=dataset_fine_subset.horizontal_coordinates,
            coarse=dataset_coarse_subset.horizontal_coordinates,
        )

        example_fine, _ = dataset_fine_subset[0]
        example_coarse, _ = dataset_coarse_subset[0]
        fine_shape = next(iter(example_fine.values())).shape[-2:]
        coarse_shape = next(iter(example_coarse.values())).shape[-2:]
        img_shape = FineResCoarseResPair(fine_shape, coarse_shape)

        fine_height, fine_width = fine_shape
        coarse_height, coarse_width = coarse_shape

        assert (
            fine_height % coarse_height == 0
        ), "Fine resolution height must be divisible by coarse resolution height"
        assert (
            fine_width % coarse_width == 0
        ), "Fine resolution width must be divisible by coarse resolution width"

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

        if requirements.use_fine_topography:
            fine_topography = get_topography(
                CoreDataLoaderConfig(
                    dataset=self.fine,
                    batch_size=self.batch_size,
                    num_data_workers=self.num_data_workers,
                )
            )
        else:
            fine_topography = None

        return GriddedData(
            dataloader,
            area_weights,
            horizontal_coordinates,
            img_shape,
            variable_metadata,
            fine_topography=fine_topography,
        )
