import warnings
from collections.abc import Sequence

import numpy as np
import xarray as xr

from fme.core.dataset.concat import XarrayConcat
from fme.core.dataset.config import (
    ConcatDatasetConfig,
    MergeDatasetConfig,
    MergeNoConcatDatasetConfig,
    RepeatedInterval,
    TimeSlice,
    XarrayDataConfig,
)
from fme.core.dataset.merged import MergedXarrayDataset
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.subset import XarraySubset
from fme.core.dataset.xarray import XarrayDataset, get_raw_paths
from fme.core.typing_ import Slice


def _as_index_selection(
    subset: Slice | TimeSlice | RepeatedInterval, dataset: XarrayDataset
) -> slice | np.ndarray:
    """Converts a subset defined either as a Slice or TimeSlice into an index slice
    based on time coordinate in provided dataset.
    """
    if isinstance(subset, Slice):
        index_selection = subset.slice
    elif isinstance(subset, TimeSlice):
        index_selection = subset.slice(dataset.sample_start_times)
    elif isinstance(subset, RepeatedInterval):
        try:
            index_selection = subset.get_boolean_mask(len(dataset), dataset.timestep)
        except ValueError as e:
            raise ValueError(f"Error when applying RepeatedInterval to dataset: {e}")
    else:
        raise TypeError(f"subset must be Slice or TimeSlice, got {type(subset)}")
    return index_selection


def get_xarray_dataset(
    config: XarrayDataConfig, names: list[str], n_timesteps: int
) -> tuple["XarraySubset", DatasetProperties]:
    dataset = XarrayDataset(config, names, n_timesteps)
    properties = dataset.properties
    index_slice = _as_index_selection(config.subset, dataset)
    dataset = XarraySubset(dataset, index_slice)
    return dataset, properties


def get_datasets(
    dataset_configs: Sequence[XarrayDataConfig],
    names: list[str],
    n_timesteps: int,
    strict: bool = True,
) -> tuple[list[XarraySubset], DatasetProperties]:
    datasets = []
    properties: DatasetProperties | None = None
    for config in dataset_configs:
        dataset, new_properties = get_xarray_dataset(config, names, n_timesteps)
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
    names: list[str],
    n_timesteps: int,
    strict: bool = True,
) -> tuple[XarrayConcat, DatasetProperties]:
    datasets, properties = get_datasets(
        dataset_configs, names, n_timesteps, strict=strict
    )
    ensemble = XarrayConcat(datasets)
    return ensemble, properties


def _infer_available_variables(config: XarrayDataConfig):
    """
    Infer the available variables from a XarrayDataset.
    """
    paths = get_raw_paths(config.data_path, config.file_pattern)
    dataset = xr.open_dataset(
        paths[0],
        decode_times=False,
        decode_timedelta=False,
        engine=config.engine,
        chunks=None,
    )
    return dataset.data_vars


def get_per_dataset_names(
    merged_config: MergeDatasetConfig | MergeNoConcatDatasetConfig,
    names: list[str],
) -> list[list[str]]:
    merged_required_names = names.copy()
    per_dataset_names = []
    for config in merged_config.merge:
        if isinstance(config, XarrayDataConfig):
            current_source_variables = _infer_available_variables(config)
        elif isinstance(config, ConcatDatasetConfig):
            current_source_variables = _infer_available_variables(config.concat[0])
        current_source_names = [
            name for name in merged_required_names if name in current_source_variables
        ]
        per_dataset_names.append(current_source_names)
        for name in current_source_names:
            merged_required_names.remove(name)
    return per_dataset_names


def get_merged_datasets(
    merged_config: MergeDatasetConfig,
    names: list[str],
    n_timesteps: int,
) -> tuple[MergedXarrayDataset, DatasetProperties]:
    merged_xarray_datasets = []
    merged_properties: DatasetProperties | None = None
    per_dataset_names = get_per_dataset_names(merged_config, names)
    config_counter = 0
    for config in merged_config.merge:
        if isinstance(config, XarrayDataConfig):
            current_source_xarray_dataset, current_source_properties = (
                get_xarray_dataset(
                    config,
                    per_dataset_names[config_counter],
                    n_timesteps,
                )
            )
            merged_xarray_datasets.append(current_source_xarray_dataset)
        elif isinstance(config, ConcatDatasetConfig):
            current_source_datasets, current_source_properties = get_datasets(
                config.concat,
                per_dataset_names[config_counter],
                n_timesteps,
                strict=config.strict,
            )
            current_source_ensemble = XarrayConcat(current_source_datasets)
            merged_xarray_datasets.append(current_source_ensemble)

        if merged_properties is None:
            merged_properties = current_source_properties
        else:
            merged_properties.update_merged_dataset(current_source_properties)
        config_counter += 1

    if merged_properties is None:
        raise ValueError("At least one dataset must be provided.")
    merged_datasets = MergedXarrayDataset(datasets=merged_xarray_datasets)
    return merged_datasets, merged_properties
