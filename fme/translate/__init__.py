"""Infrastructure for domain-translation and multi-resolution training.

This package sits at the coupled/downscaling tier: it may import ``fme.core``,
``fme.ace``, and ``fme.downscaling``; nothing imports it. It provides the
shared abstraction of a named pool of components (trainable transforms /
encoders / decoders wrapped around a backbone stepper) mapping between named
domains, used by the transfer learning and multi-resolution latent-stepping
programs.
"""

from .components import (
    BackboneConfig,
    ComponentPool,
    ComponentPoolConfig,
    TransformConfig,
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
    "SameGridTransformConfig",
    "TransformConfig",
    "TransformModuleConfig",
    "TransformSelector",
]
