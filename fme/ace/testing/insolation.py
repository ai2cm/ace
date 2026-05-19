import contextlib
import unittest.mock

import torch

from fme.ace.stepper.insolation import CM4Insolation


@contextlib.contextmanager
def patch_cm4_solar_constant(value: float = 1.0):
    """Temporarily force ``CM4Insolation`` to use a fixed solar constant.

    Useful in tests where the configured solar constant (literal or loaded
    from disk) produces insolation magnitudes that are inconsistent with the
    on-disk normalization stats, leading to NaN training losses.
    """
    original_call = CM4Insolation.__call__

    def patched_call(self, time, timestep, horizontal_coordinates, solar_constant):
        replacement = torch.as_tensor(
            value, dtype=solar_constant.dtype, device=solar_constant.device
        )
        return original_call(self, time, timestep, horizontal_coordinates, replacement)

    with unittest.mock.patch.object(CM4Insolation, "__call__", patched_call):
        yield
