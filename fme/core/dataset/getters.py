import warnings
from typing import List, Mapping, Optional, Sequence, Tuple

from fme.core.dataset.config import XarrayDataConfig
from fme.core.dataset.xarray import (
    DatasetProperties,
    MergedXarrayDataset,
    XarrayConcat,
    XarraySubset,
    get_merged_requirements,
    get_xarray_dataset,
)

from .requirements import DataRequirements


def get_datasets(
    dataset_configs: Sequence[XarrayDataConfig],
    requirements: DataRequirements,
    strict: bool = True,
) -> Tuple[List[XarraySubset], DatasetProperties]:
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
) -> Tuple[XarrayConcat, DatasetProperties]:
    datasets, properties = get_datasets(dataset_configs, requirements, strict=strict)
    ensemble = XarrayConcat(datasets)
    return ensemble, properties


def get_merged_datasets(
    dataset_configs: Mapping[str, Sequence[XarrayDataConfig]],
    requirements: DataRequirements,
    strict: bool = True,
) -> Tuple[MergedXarrayDataset, DatasetProperties]:
    merged_xarray_datasets = []
    merged_properties: Optional[DatasetProperties] = None
    merged_requirements = get_merged_requirements(dataset_configs, requirements)
    for key, config in dataset_configs.items():
        current_source_datasets, current_source_properties = get_datasets(
            config,
            merged_requirements[key],
            strict=strict,
        )
        current_source_ensemble = XarrayConcat(current_source_datasets)
        merged_xarray_datasets.append(current_source_ensemble)
        if merged_properties is None:
            merged_properties = current_source_properties
        else:
            merged_properties.update_merged_dataset(current_source_properties)

    if merged_properties is None:
        raise ValueError("At least one dataset must be provided.")
    merged_datasets = MergedXarrayDataset(datasets=merged_xarray_datasets)
    return merged_datasets, merged_properties
