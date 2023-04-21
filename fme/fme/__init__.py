from .core.version import __version__

__version__ = "0.1.0"

from .core.metrics import (
    lat_cell_centers,
    spherical_area_weights,
    weighted_mean,
    weighted_mean_bias,
    root_mean_squared_error,
)

__all__ = [
    'lat_cell_centers',
    'spherical_area_weights',
    'weighted_mean',
    'weighted_mean_bias',
    'root_mean_squared_error',
]