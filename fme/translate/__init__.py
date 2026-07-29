"""Infrastructure for domain-translation and multi-resolution training.

This package sits at the coupled/downscaling tier: it may import ``fme.core``,
``fme.ace``, and ``fme.downscaling``; nothing imports it. It provides the
shared abstraction of a named pool of components (trainable transforms /
encoders / decoders wrapped around a backbone stepper) mapping between named
domains, used by the transfer learning and multi-resolution latent-stepping
programs, plus the named data streams (:mod:`fme.translate.data`) that pair with
those domains.
"""

from .components import (
    BackboneConfig,
    ComponentPool,
    ComponentPoolConfig,
    TransformConfig,
)
from .data import (
    ObjectiveDataRequirements,
    StreamConfig,
    StreamRequirements,
    TranslateBatchData,
    TranslateDataLoaderConfig,
    TranslateDataRequirements,
    TranslateGriddedData,
    get_gridded_data,
)
from .domains import DomainConfig, LatentChannels
from .modules import (
    InterpolateTransformConfig,
    SameGridTransformConfig,
    TransformModuleConfig,
    TransformSelector,
)

__all__ = [
    "BackboneConfig",
    "ComponentPool",
    "ComponentPoolConfig",
    "DomainConfig",
    "InterpolateTransformConfig",
    "LatentChannels",
    "ObjectiveDataRequirements",
    "SameGridTransformConfig",
    "StreamConfig",
    "StreamRequirements",
    "TransformConfig",
    "TransformModuleConfig",
    "TransformSelector",
    "TranslateBatchData",
    "TranslateDataLoaderConfig",
    "TranslateDataRequirements",
    "TranslateGriddedData",
    "get_gridded_data",
]
