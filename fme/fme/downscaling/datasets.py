"""Contains code relating to loading (highres, lowres) examples for downscaling."""

import dataclasses
from typing import List, Literal, Mapping, Optional, Sequence, Tuple

import torch
import torch.utils.data
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import fme.core.data_loading.params
from fme.core.data_loading._xarray import XarrayDataset
from fme.core.data_loading.data_typing import HorizontalCoordinates
from fme.core.data_loading.requirements import DataRequirements
from fme.core.device import using_gpu
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping
from fme.downscaling.typing_ import HighResLowResPair


@dataclasses.dataclass
class BatchData:
    highres: Mapping[str, torch.Tensor]
    lowres: Mapping[str, torch.Tensor]
    times: xr.DataArray

    @classmethod
    def from_sample_tuples(
        cls,
        samples: Sequence[Tuple[TensorMapping, TensorMapping, xr.DataArray]],
        sample_dim_name: str = "sample",
    ) -> "BatchData":
        highres, lowres, times = zip(*samples)
        return cls(
            torch.utils.data.default_collate(highres),
            torch.utils.data.default_collate(lowres),
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
class DownscalingDataLoader:
    loader: torch.utils.data.DataLoader
    area_weights: HighResLowResPair[torch.Tensor]
    horizontal_coordinates: HighResLowResPair[HorizontalCoordinates]
    img_shape: HighResLowResPair[Tuple[int, int]]


@dataclasses.dataclass
class DataLoaderParams:
    path_highres: str
    path_lowres: str
    data_type: Literal["xarray", "ensemble_xarray"]
    batch_size: int
    num_data_workers: int

    def build(
        self, train: bool, var_names: List[str], dist: Optional[Distributed] = None
    ) -> DownscalingDataLoader:
        if dist is None:
            dist = Distributed.get_instance()

        dataset_highres, dataset_lowres = [
            XarrayDataset(
                fme.core.data_loading.params.DataLoaderParams(
                    dataset=fme.core.data_loading.params.XarrayDataParams(
                        data_path=path,
                    ),
                    data_type=self.data_type,
                    batch_size=self.batch_size,
                    num_data_workers=self.num_data_workers,
                ),
                DataRequirements(var_names, 1),
            )
            for path in (self.path_highres, self.path_lowres)
        ]

        dataset = PairedDataset(dataset_highres, dataset_lowres)

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

        area_weights = HighResLowResPair(
            dataset_highres.area_weights, dataset_lowres.area_weights
        )
        horizontal_coordinates = HighResLowResPair(
            highres=dataset_highres.horizontal_coordinates,
            lowres=dataset_lowres.horizontal_coordinates,
        )

        example_highres, _ = dataset_highres[0]
        example_lowres, _ = dataset_lowres[0]
        highres_shape = next(iter(example_highres.values())).shape[-2:]
        lowres_shape = next(iter(example_lowres.values())).shape[-2:]
        img_shape = HighResLowResPair(highres_shape, lowres_shape)

        return DownscalingDataLoader(
            dataloader, area_weights, horizontal_coordinates, img_shape
        )
