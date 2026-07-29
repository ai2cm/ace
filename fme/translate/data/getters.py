"""Building a :class:`TranslateGriddedData` from a config and the objectives'
requirements.

The shape mirrors :func:`fme.ace.data_loading.getters.get_gridded_data`: build
the datasets, get a distributed sampler, hand both to a data loader, wrap it.
What is multi-stream about it is that there is one dataset per stream, one
paired dataset and loader per pairing group, and a distinct sampler seed per
group.
"""

import logging
from collections.abc import Mapping

from fme.core.dataset.dataset import DatasetABC
from fme.core.dataset.properties import DatasetProperties
from fme.core.device import using_gpu
from fme.core.distributed import Distributed
from fme.core.labels import LabelEncoding
from fme.translate.data.batch_data import TranslateCollateFn
from fme.translate.data.config import TranslateDataLoaderConfig
from fme.translate.data.dataloader import TranslateDataLoader
from fme.translate.data.dataset import PairedStreamDataset
from fme.translate.data.gridded_data import TranslateGriddedData
from fme.translate.data.requirements import TranslateDataRequirements

__all__ = ["get_gridded_data"]

logger = logging.getLogger(__name__)


def _validate_streams_match_requirements(
    config: TranslateDataLoaderConfig, requirements: TranslateDataRequirements
) -> None:
    configured = set(config.stream_configs)
    required = set(requirements.streams)
    unknown = sorted(required - configured)
    if unknown:
        raise ValueError(
            f"The objectives reference data streams that are not configured: "
            f"{unknown}. Configured streams are {sorted(configured)}."
        )
    unused = sorted(configured - required)
    if unused:
        raise ValueError(
            f"These data streams are configured but no objective consumes them: "
            f"{unused}. Remove them, or check the stream names the objectives "
            "reference."
        )


def _build_group_loader(
    config: TranslateDataLoaderConfig,
    datasets: Mapping[str, DatasetABC],
    label_encodings: Mapping[str, LabelEncoding | None],
    properties: Mapping[str, DatasetProperties],
    train: bool,
    group_index: int,
    n_groups: int,
) -> TranslateDataLoader:
    dataset = PairedStreamDataset(datasets)
    dist = Distributed.get_instance()
    # Each group gets its own seed offset so equal-length groups are not handed
    # the same permutation, which would silently index-pair streams that are
    # meant to be sampled independently.
    sampler = dist.get_sampler(
        dataset.torch_dataset, shuffle=train, seed_offset=group_index
    )

    if config.zarr_engine_used and config.num_data_workers > 0:
        # GCSFS and S3FS are not fork-safe, and reading zarr with async from
        # weka also requires forkserver.
        mp_context: str | None = "forkserver"
        persistent_workers = True
        dataset.enable_shared_memory()
    else:
        mp_context = None
        persistent_workers = False

    # DataLoader's own default for prefetch_factor is not None, so it must be
    # left unset rather than passed as None.
    kwargs = (
        {}
        if config.prefetch_factor is None
        else {"prefetch_factor": config.prefetch_factor}
    )
    loader = TranslateDataLoader(
        dataset=dataset,
        collate_fn=TranslateCollateFn(
            horizontal_dims={
                name: list(properties[name].horizontal_coordinates.dims)
                for name in datasets
            },
            label_encodings={name: label_encodings[name] for name in datasets},
        ),
        sampler=sampler,
        batch_size=dist.local_batch_size(int(config.batch_size)),
        num_workers=config.num_data_workers,
        drop_last=True,
        pin_memory=using_gpu(),
        multiprocessing_context=mp_context,
        persistent_workers=persistent_workers,
        **kwargs,
    )
    if len(loader) == 0:
        raise ValueError(
            f"No batches in the loader for pairing group {group_index} of "
            f"{n_groups} (streams {sorted(datasets)}): {loader.n_samples} "
            f"samples at batch size {loader.batch_size}."
        )
    return loader


def get_gridded_data(
    config: TranslateDataLoaderConfig,
    requirements: TranslateDataRequirements,
    train: bool,
) -> TranslateGriddedData:
    """Build the multi-stream data loader.

    Args:
        config: The configured data streams and batching.
        requirements: The merged objective requirements, which decide what each
            stream loads and which streams share a time index.
        train: Whether the loader is for training rather than validation data;
            if True, samples are shuffled.

    Returns:
        Data over every configured stream, one batch carrying all of them.
    """
    _validate_streams_match_requirements(config, requirements)
    stream_configs = config.stream_configs
    datasets: dict[str, DatasetABC] = {}
    properties: dict[str, DatasetProperties] = {}
    label_encodings: dict[str, LabelEncoding | None] = {}
    for name, stream in stream_configs.items():
        stream_requirements = requirements.streams[name]
        dataset, stream_properties = stream.get_dataset(
            stream_requirements.names, stream_requirements.n_timesteps_schedule
        )
        datasets[name] = dataset
        properties[name] = stream_properties
        available_labels = stream.dataset.available_labels
        label_encodings[name] = (
            None
            if available_labels is None
            else LabelEncoding(sorted(available_labels))
        )

    groups = requirements.pairing_groups
    group_loaders = [
        _build_group_loader(
            config=config,
            datasets={name: datasets[name] for name in group},
            label_encodings=label_encodings,
            properties=properties,
            train=train,
            group_index=group_index,
            n_groups=len(groups),
        )
        for group_index, group in enumerate(groups)
    ]
    return TranslateGriddedData(
        group_loaders=group_loaders,
        stream_properties=properties,
        stream_domains={name: stream.domain for name, stream in stream_configs.items()},
    )
