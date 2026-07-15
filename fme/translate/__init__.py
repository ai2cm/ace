"""Infrastructure for domain-translation and multi-resolution training.

This package sits at the coupled/downscaling tier: it may import ``fme.core``,
``fme.ace``, and ``fme.downscaling``; nothing imports it. It provides the
shared abstraction of a named pool of components (trainable transforms /
encoders / decoders wrapped around a backbone stepper) used by the transfer
learning and multi-resolution latent-stepping programs.
"""

from .components import (
    BackboneConfig,
    ComponentPool,
    ComponentPoolConfig,
    TransformConfig,
)

__all__ = [
    "BackboneConfig",
    "ComponentPool",
    "ComponentPoolConfig",
    "TransformConfig",
]
