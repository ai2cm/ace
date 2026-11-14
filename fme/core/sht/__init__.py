"""Centralized SHT (Spherical Harmonic Transform) package.

All SHT-related classes and helpers live here.  Consumers inside ``fme/``
should ``from fme.core.sht import ...`` rather than importing from
``torch_harmonics`` directly.
"""

# Upstream torch_harmonics utilities that consumers need
from torch_harmonics import ResampleS2, quadrature
from torch_harmonics.quadrature import (
    clenshaw_curtiss_weights,
    legendre_gauss_weights,
    lobatto_weights,
)

# Upstream distributed wrappers (re-exports from torch_harmonics.distributed)
from fme.core.sht._distributed import (
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

# FFT helpers (forked from torch_harmonics)
from fme.core.sht._fft import irfft, rfft

# Forked transform classes (patched RealSHT / InverseRealSHT from torch_harmonics)
from fme.core.sht._transforms import (
    InverseRealSHT,
    InverseRealSHTBenchmark,
    RealSHT,
    RealSHTBenchmark,
)

__all__ = [
    # Transforms
    "RealSHT",
    "InverseRealSHT",
    "RealSHTBenchmark",
    "InverseRealSHTBenchmark",
    # Distributed
    "DistributedRealSHT",
    "DistributedInverseRealSHT",
    "DistributedDiscreteContinuousConvS2",
    "DistributedResampleS2",
    "compute_split_shapes",
    "distributed_transpose_azimuth",
    "distributed_transpose_polar",
    "init",
    "is_initialized",
    # FFT
    "rfft",
    "irfft",
    # Upstream torch_harmonics resampling
    "ResampleS2",
    # Quadrature
    "quadrature",
    "legendre_gauss_weights",
    "lobatto_weights",
    "clenshaw_curtiss_weights",
]
