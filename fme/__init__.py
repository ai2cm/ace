__version__ = "2025.7.0"


from . import ace
from . import sht_fix as _sht_fix
from .core import Packer, StandardNormalizer, get_device, get_normalizer, using_gpu
from .core.metrics import (
    gradient_magnitude,
    gradient_magnitude_percent_diff,
    rmse_of_time_mean,
    root_mean_squared_error,
    spherical_area_weights,
    time_and_global_mean_bias,
    weighted_mean,
    weighted_mean_bias,
    weighted_mean_gradient_magnitude,
    weighted_nanmean,
    weighted_std,
    weighted_sum,
)

__all__ = [
    "spherical_area_weights",
    "weighted_mean",
    "weighted_mean_bias",
    "root_mean_squared_error",
    "gradient_magnitude",
    "weighted_mean_gradient_magnitude",
    "rmse_of_time_mean",
    "time_and_global_mean_bias",
    "gradient_magnitude_percent_diff",
    "get_device",
    "get_normalizer",
    "Packer",
    "StandardNormalizer",
    "using_gpu",
    "weighted_std",
    "ace",
]
