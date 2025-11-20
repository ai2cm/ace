import logging

import torch.utils.data

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.data_loading.dataloader import get_data_loader
from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements
from fme.core.dataset.merged import MergeNoConcatDatasetConfig
from fme.core.dataset.xarray import XarrayDataConfig, XarrayDataset
from fme.core.device import using_gpu
from fme.core.distributed import Distributed

from .batch_data import PrognosticState
from .config import DataLoaderConfig
from .gridded_data import GriddedData, InferenceGriddedData
from .inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceDataLoaderConfig,
    InferenceDataset,
)

logger = logging.getLogger(__name__)

class CollateFn:
    def __init__(self, horizontal_dims: list[str]):
        self.horizontal_dims = horizontal_dims

    def __call__(self, samples):
        return BatchData.from_sample_tuples(
            samples,
            horizontal_dims=self.horizontal_dims,
        )


def _get_sampler(
    dataset: torch.utils.data.Dataset,
    sample_with_replacement_dataset_size: int | None,
    train: bool,
) -> torch.utils.data.Sampler:
    dist = Distributed.get_instance()
    if sample_with_replacement_dataset_size is not None:
        local_sample_with_replacement_dataset_size = (
            sample_with_replacement_dataset_size // dist.world_size
        )
        sampler = torch.utils.data.RandomSampler(
            dataset,
            num_samples=local_sample_with_replacement_dataset_size,
            replacement=True,
        )
    else:
        sampler = dist.get_sampler(dataset, shuffle=train)
    return sampler


def get_gridded_data(
    config: DataLoaderConfig,
    train: bool,
    requirements: DataRequirements,
) -> GriddedData:
    """
    Args:
        config: Parameters for the data loader.
        train: Whether loader is intended for training or validation data; if True,
            then data will be shuffled.
        requirements: Data requirements for the model.
    """
    n_timesteps_preloaded = config.time_buffer + requirements.n_timesteps
    dataset, properties = config.get_dataset(requirements.names, n_timesteps_preloaded)

    if config.time_buffer > 0:
        # include requirements.n_timesteps - 1 steps of overlap so that no samples are
        # skipped at the boundaries of the preloaded timesteps
        start_every_n = config.time_buffer + 1
        indices = range(len(dataset))[::start_every_n]
        dataset = torch.utils.data.Subset(dataset, indices)

    dist = Distributed.get_instance()

    sampler = _get_sampler(dataset, config.sample_with_replacement, train)

    if config.zarr_engine_used:
        # GCSFS and S3FS are not fork-safe, so we need to use forkserver
        # reading zarr with async from weka also requires forkserver
        mp_context = "forkserver"
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    dist = Distributed.get_instance()
    batch_size = dist.local_batch_size(int(config.batch_size))

    dataloader = get_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        n_window_timesteps=requirements.n_timesteps,
        time_buffer=config.time_buffer,
        num_workers=config.num_data_workers,
        sampler=sampler,
        shuffled=train,
        drop_last=True,
        pin_memory=using_gpu(),
        collate_fn=CollateFn(list(properties.horizontal_coordinates.dims)),
        multiprocessing_context=mp_context,
        persistent_workers=persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )

    return GriddedData(
        loader=dataloader,
        properties=properties,
        sampler=sampler,
        modifier=config.augmentation.build_modifier(),
    )


def get_inference_data(
    config: InferenceDataLoaderConfig,
    total_forward_steps: int,
    window_requirements: DataRequirements,
    initial_condition: PrognosticState | PrognosticStateDataRequirements,
    label_override: list[str] | None = None,
    surface_temperature_name: str | None = None,
    ocean_fraction_name: str | None = None,
) -> InferenceGriddedData:
    """
    Args:
        config: Parameters for the data loader.
        total_forward_steps: Total number of forward steps to take over the course of
            inference.
        window_requirements: Data requirements for the model.
        initial_condition: Initial condition for the inference, or a requirements object
            specifying how to extract the initial condition from the first batch of
            data
        label_override: Labels for the forcing data to be provided on each sample
            instead of the labels in the dataset.
        surface_temperature_name: Name of the surface temperature variable. Can be
            set to None if no ocean temperature prescribing is being used.
        ocean_fraction_name: Name of the ocean fraction variable. Can be set to None
            if no ocean temperature prescribing is being used.

    Returns:
        A data loader for inference with coordinates and metadata.
    """
    dataset = InferenceDataset(
        config=config,
        total_forward_steps=total_forward_steps,
        requirements=window_requirements,
        surface_temperature_name=surface_temperature_name,
        ocean_fraction_name=ocean_fraction_name,
        label_override=label_override,
    )
    properties = dataset.properties

    if config.zarr_engine_used:
        # GCSFS and S3FS are not fork-safe, so we need to use forkserver
        # persist workers since startup is slow
        mp_context = "forkserver"
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    logging.info(f"Multiprocessing inference context: {mp_context or 'fork'}")

    # we roll our own batching in InferenceDataset, which is why batch_size=None below
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_data_workers,
        shuffle=False,
        pin_memory=using_gpu(),
        multiprocessing_context=mp_context,
        persistent_workers=persistent_workers,
    )
    gridded_data = InferenceGriddedData(
        loader=loader,
        initial_condition=initial_condition,
        properties=properties,
    )

    return gridded_data


def get_forcing_data(
    config: ForcingDataLoaderConfig,
    total_forward_steps: int,
    window_requirements: DataRequirements,
    initial_condition: PrognosticState,
    surface_temperature_name: str | None = None,
    ocean_fraction_name: str | None = None,
    label_override: list[str] | None = None,
) -> InferenceGriddedData:
    """Return a GriddedData loader for forcing data based on the initial condition.
    This function determines the start indices for the forcing data based on the initial
    time in the provided initial condition.

    Args:
        config: Parameters for the forcing data loader.
        total_forward_steps: Total number of forward steps to take over the course of
            inference.
        window_requirements: Data requirements for the forcing data.
        initial_condition: Initial condition for the inference.
        surface_temperature_name: Name of the surface temperature variable. Can be
            set to None if no ocean temperature prescribing is being used.
        ocean_fraction_name: Name of the ocean fraction variable. Can be set to None
            if no ocean temperature prescribing is being used.
        label_override: Labels for the forcing data. If provided, these labels will be
            provided on each sample instead of the labels in the dataset.

    Returns:
        A data loader for forcing data with coordinates and metadata.
    """
    initial_time = initial_condition.as_batch_data().time
    if initial_time.shape[1] != 1:
        raise NotImplementedError("code assumes initial time only has 1 timestep")
    if isinstance(config.dataset, XarrayDataConfig):
        available_times = XarrayDataset(
            config.dataset, window_requirements.names, window_requirements.n_timesteps
        ).all_times
    elif isinstance(config.dataset, MergeNoConcatDatasetConfig):
        # Some forcing variables may not be in the first dataset,
        # use an empty data requirements to get all times
        if isinstance(config.dataset.merge[0], XarrayDataConfig):
            available_times = XarrayDataset(
                config.dataset.merge[0],
                names=[],
                n_timesteps=window_requirements.n_timesteps,
            ).all_times
        else:
            raise ValueError("Forcing data cannot be concatenated.")
    start_time_indices = []
    for time in initial_time.values[:, 0]:
        start_time_indices.append(available_times.get_loc(time))
    inference_config = config.build_inference_config(
        start_indices=ExplicitIndices(start_time_indices)
    )
    return get_inference_data(
        config=inference_config,
        total_forward_steps=total_forward_steps,
        window_requirements=window_requirements,
        initial_condition=initial_condition,
        surface_temperature_name=surface_temperature_name,
        ocean_fraction_name=ocean_fraction_name,
        label_override=label_override,
    )
