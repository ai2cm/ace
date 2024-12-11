import abc
from collections import namedtuple
from typing import Tuple

import torch
import xarray as xr

from fme.core.typing_ import TensorDict

VariableMetadata = namedtuple("VariableMetadata", ["units", "long_name"])


class Dataset(torch.utils.data.Dataset, abc.ABC):
    @abc.abstractmethod
    def get_sample_by_time_slice(
        self, time_slice: slice
    ) -> Tuple[TensorDict, xr.DataArray]:
        """
        Returns a sample of data for the given time slice.

        Args:
            time_slice: The time slice to return data for.

        Returns:
            A tuple whose first item is a mapping from variable
            name to tensor of shape [n_time, n_lat, n_lon] and
            whose second item is a time coordinate array.
        """
        ...
