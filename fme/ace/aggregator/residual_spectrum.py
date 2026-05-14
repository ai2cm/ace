"""Residual (tendency) power spectra via the spherical harmonic diagnostic.

Computes temporal differences from consecutive timesteps in the data,
then feeds those tendency fields to ``PairedSphericalPowerSpectrumAggregator``
for spectral analysis.  This reveals whether the model's tendency fields
lose power at small scales (spectral smoothing) relative to the target.
"""

import torch

from fme.core.typing_ import TensorMapping


def temporal_diffs(data: TensorMapping) -> TensorMapping:
    """Compute signed temporal differences along axis 1.

    Returns a new mapping with the same keys, where each tensor's time
    dimension is reduced by one: ``out[name] = data[name][:, 1:] - data[name][:, :-1]``.
    Variables with fewer than 2 timesteps are dropped.
    """
    result: dict[str, torch.Tensor] = {}
    for name, tensor in data.items():
        if tensor.shape[1] >= 2:
            result[name] = tensor[:, 1:] - tensor[:, :-1]
    return result
