import warnings
from typing import List, Optional, Sequence, Tuple

import torch.utils.data

from fme.core.dataset.config import XarrayDataConfig
from fme.core.dataset.xarray import DatasetProperties, XarrayDataset, get_xarray_dataset

from .requirements import DataRequirements


def get_datasets(
    dataset_configs: Sequence[XarrayDataConfig],
    requirements: DataRequirements,
    strict: bool = True,
) -> Tuple[List[XarrayDataset], DatasetProperties]:
    datasets = []
    properties: Optional[DatasetProperties] = None
    for config in dataset_configs:
        dataset, new_properties = get_xarray_dataset(config, requirements)
        datasets.append(dataset)
        if properties is None:
            properties = new_properties
        elif not strict:
            try:
                properties.update(new_properties)
            except ValueError as e:
                warnings.warn(
                    f"Metadata for each ensemble member are not the same: {e}"
                )
        else:
            properties.update(new_properties)
    if properties is None:
        raise ValueError("At least one dataset must be provided.")

    return datasets, properties


def get_dataset(
    dataset_configs: Sequence[XarrayDataConfig],
    requirements: DataRequirements,
    strict: bool = True,
) -> Tuple[torch.utils.data.ConcatDataset[XarrayDataset], DatasetProperties]:
    datasets, properties = get_datasets(dataset_configs, requirements, strict=strict)
    ensemble = torch.utils.data.ConcatDataset(datasets)
    return ensemble, properties
