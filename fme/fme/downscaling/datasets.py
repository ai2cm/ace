"""Contains code relating to loading (fine, coarse) examples for downscaling."""

import dataclasses
from typing import Literal, Mapping, Optional, Sequence, Tuple

import torch
import torch.utils.data
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import fme.core.data_loading.config
from fme.core.data_loading.data_typing import HorizontalCoordinates, VariableMetadata
from fme.core.data_loading.getters import get_dataset
from fme.core.data_loading.requirements import DataRequirements
from fme.core.device import using_gpu
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping
from fme.downscaling.typing_ import FineResCoarseResPair


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
        # TODO(gideond) Note that this check may not hold as we focus on precip outputs
        assert batch1.keys() == batch2.keys(), "Examples must have the same variables."
        assert all(times1 == times2), "Times must match."

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        batch1, times1 = self.dataset1[idx]
        batch2, _ = self.dataset2[idx]
        return batch1, batch2, times1


@dataclasses.dataclass
class GriddedData:
    loader: torch.utils.data.DataLoader
    area_weights: FineResCoarseResPair[torch.Tensor]
    horizontal_coordinates: FineResCoarseResPair[HorizontalCoordinates]
    img_shape: FineResCoarseResPair[Tuple[int, int]]
    metadata: Mapping[str, VariableMetadata]

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
    path_fine: str
    path_coarse: str
    data_type: Literal["xarray", "ensemble_xarray"]
    batch_size: int
    num_data_workers: int

    def build(
        self,
        train: bool,
        requirements: DataRequirements,
        dist: Optional[Distributed] = None,
    ) -> GriddedData:
        if dist is None:
            dist = Distributed.get_instance()

        dataset_fine, dataset_coarse = [
            get_dataset(
                fme.core.data_loading.config.DataLoaderConfig(
                    dataset=fme.core.data_loading.config.XarrayDataConfig(
                        data_path=path,
                    ),
                    data_type=self.data_type,
                    batch_size=self.batch_size,
                    num_data_workers=self.num_data_workers,
                ),
                requirements=requirements,
            )
            for path in (self.path_fine, self.path_coarse)
        ]

        dataset = PairedDataset(dataset_fine, dataset_coarse)

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
            dataset_fine.area_weights, dataset_coarse.area_weights
        )
        horizontal_coordinates = FineResCoarseResPair(
            fine=dataset_fine.horizontal_coordinates,
            coarse=dataset_coarse.horizontal_coordinates,
        )

        example_fine, _ = dataset_fine[0]
        example_coarse, _ = dataset_coarse[0]
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

        assert dataset_fine.metadata == dataset_coarse.metadata, "Metadata must match."
        metadata = dataset_fine.metadata

        return GriddedData(
            dataloader,
            area_weights,
            horizontal_coordinates,
            img_shape,
            metadata,
        )
