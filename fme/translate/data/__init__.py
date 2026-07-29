"""Named data streams and the multi-stream loader that samples them.

A *stream* is one named source of data bound to one component-pool domain (see
:mod:`fme.translate.domains`). Several streams may serve one domain — ERA5 and
IFS both feeding ``atmos_1deg`` — so the stream ``name`` (the handle objectives
reference) is distinct from the ``domain`` it serves, and defaults to it.

Which streams are sampled at the same valid times is *derived*, never
configured: the objectives declare what each of them needs
(:class:`ObjectiveDataRequirements`), and
:meth:`TranslateDataRequirements.from_objectives` merges those into per-stream
:class:`fme.ace.requirements.DataRequirements` plus the *pairing groups* —
connected components of the graph in which every objective ties together the
streams it consumes. Streams in a group are sampled at the same valid times;
groups are sampled independently of one another. A sampling knob in the config
could contradict the objectives; a derivation cannot.
"""

from .batch_data import TranslateBatchData, TranslateCollateFn
from .config import StreamConfig, TranslateDataLoaderConfig
from .dataloader import TranslateDataLoader
from .dataset import PairedStreamDataset
from .getters import get_gridded_data
from .gridded_data import TranslateGriddedData
from .requirements import (
    ObjectiveDataRequirements,
    StreamRequirements,
    TranslateDataRequirements,
)

__all__ = [
    "ObjectiveDataRequirements",
    "PairedStreamDataset",
    "StreamConfig",
    "StreamRequirements",
    "TranslateBatchData",
    "TranslateCollateFn",
    "TranslateDataLoader",
    "TranslateDataLoaderConfig",
    "TranslateDataRequirements",
    "TranslateGriddedData",
    "get_gridded_data",
]
