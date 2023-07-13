from typing import Mapping, Optional, Dict
from collections import namedtuple
import torch
from netCDF4 import Dataset
import numpy as np


class DataWriter:
    """
    Write data to a netCDF file.
    """

    VariableMetadata = namedtuple("VariableMetadata", ["units", "long_name"])

    def __init__(
        self, filename: str, n_samples: int, metadata: Mapping[str, VariableMetadata]
    ):
        """
        Args:
            filename: Path to the netCDF file to write to.
            n_samples: Number of samples to write to the file.
            metadata: Metadata for each variable to be written to the file.
        """
        self.filename = filename
        self.metadata = metadata
        self.dataset = Dataset(filename, "w", format="NETCDF4")
        self.dataset.createDimension("source", 2)
        self.dataset.createDimension("timestep", None)  # unlimited dimension
        self.dataset.createDimension("sample", n_samples)
        self.dataset.createVariable("source", "str", ("source",))
        self.dataset.variables["source"][:] = np.array(["target", "prediction"])
        self._n_lat: Optional[int] = None
        self._n_lon: Optional[int] = None

    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        start_sample: int,
    ):
        """
        Append a batch of data to the file.

        Args:
            target: Target data.
            prediction: Prediction data.
            start_timestep: Timestep at which to start writing.
            start_sample: Sample at which to start writing.
        """
        if self._n_lat is None:
            self._n_lat = target[next(iter(target.keys()))].shape[-2]
            self.dataset.createDimension("n_lat", self._n_lat)
        if self._n_lon is None:
            self._n_lon = target[next(iter(target.keys()))].shape[-1]
            self.dataset.createDimension("n_lon", self._n_lon)

        dims = ("source", "sample", "timestep", "n_lat", "n_lon")
        for variable_name in set(target.keys()).union(prediction.keys()):
            if variable_name not in self.dataset.variables:
                self.dataset.createVariable(
                    variable_name,
                    "f4",
                    dims,
                    fill_value=np.nan,
                )
                if variable_name in self.metadata:
                    self.dataset.variables[variable_name].units = self.metadata[
                        variable_name
                    ].units
                    self.dataset.variables[variable_name].long_name = self.metadata[
                        variable_name
                    ].long_name

        if variable_name in target:
            target_numpy = target[variable_name].cpu().numpy()
        else:
            target_numpy = np.full(
                shape=target[next(iter(target.keys()))].shape, fill_value=np.nan
            )
        if variable_name in prediction:
            prediction_numpy = prediction[variable_name].cpu().numpy()
        else:
            prediction_numpy = np.full(
                shape=prediction[next(iter(prediction.keys()))].shape, fill_value=np.nan
            )

        # Broadcast the corresponding dimension to match with the
        # 'source' dimension of the variable in the netCDF file
        target_numpy = np.expand_dims(target_numpy, dims.index("source"))
        prediction_numpy = np.expand_dims(prediction_numpy, dims.index("source"))

        time_len = self.dataset.variables[variable_name].shape[1]
        if start_sample + target_numpy.shape[1] > time_len:
            raise ValueError(
                f"Batch size {target_numpy.shape[1]} starting at sample {start_sample} "
                "is too large to fit in the netCDF file with time "
                f"dimension of length {time_len}."
            )
        # Append the data to the variables, this assignment writes directly to disk
        self.dataset.variables[variable_name][
            :,
            start_sample : start_sample + target_numpy.shape[1],
            start_timestep : start_timestep + target_numpy.shape[2],
            :,
        ] = np.concatenate([target_numpy, prediction_numpy], axis=0)


class NullDataWriter:
    """
    Null pattern for DataWriter, which does nothing.
    """

    VariableMetadata = namedtuple("VariableMetadata", ["units", "long_name"])

    def __init__(self):
        pass

    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        start_sample: int,
    ):
        pass
