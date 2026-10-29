"""Thin re-exports of distributed SHT helpers from torch_harmonics.distributed.

Consumers should import these from ``fme.core.sht`` rather than reaching into
``torch_harmonics.distributed`` directly, so that we have a single place to
swap in patched subclasses (e.g. PR 1602's ``_sht_patch.py``).
"""

from torch_harmonics.distributed import (
    DistributedDiscreteContinuousConvS2,
    DistributedInverseRealSHT,
    DistributedRealSHT,
    DistributedResampleS2,
    compute_split_shapes,
    distributed_transpose_azimuth,
    distributed_transpose_polar,
    init,
    is_initialized,
)

__all__ = [
    "DistributedRealSHT",
    "DistributedInverseRealSHT",
    "DistributedDiscreteContinuousConvS2",
    "DistributedResampleS2",
    "compute_split_shapes",
    "distributed_transpose_azimuth",
    "distributed_transpose_polar",
    "init",
    "is_initialized",
]
