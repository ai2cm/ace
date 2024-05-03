import dataclasses
from typing import Dict, Protocol, Tuple

import torch
import xarray as xr

TIME_DIM_NAME = "time"
TIME_DIM = 1  # sample, time, lat, lon


class _DataWriter(Protocol):
    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        start_sample: int,
        batch_times: xr.DataArray,
    ):
        pass

    def flush(self):
        pass


@dataclasses.dataclass
class TimeCoarsenConfig:
    """
    Config for inference data time coarsening.

    Args:
        coarsen_factor: Factor by which to coarsen in time, an integer 1 or greater. The
            resulting time labels will be coarsened to the mean of the original labels.
    """

    def __post_init__(self):
        if self.coarsen_factor < 1:
            raise ValueError(
                f"coarsen_factor must be 1 or greater, got {self.coarsen_factor}"
            )

    coarsen_factor: int

    def build(self, data_writer: _DataWriter) -> "TimeCoarsen":
        return TimeCoarsen(
            data_writer=data_writer,
            coarsen_factor=self.coarsen_factor,
        )

    def n_coarsened_timesteps(self, n_timesteps: int) -> int:
        """Assumes initial condition is NOT in n_timesteps"""
        return (n_timesteps) // self.coarsen_factor


class TimeCoarsen:
    """Wraps a data writer and coarsens its arguments in time before passing them on."""

    def __init__(
        self,
        data_writer: _DataWriter,
        coarsen_factor: int,
    ):
        self._data_writer: _DataWriter = data_writer
        self._coarsen_factor: int = coarsen_factor

    def append_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        start_sample: int,
        batch_times: xr.DataArray,
    ):
        (
            target_coarsened,
            prediction_coarsened,
            start_timestep,
            batch_times_coarsened,
        ) = self.coarsen_batch(target, prediction, start_timestep, batch_times)
        self._data_writer.append_batch(
            target_coarsened,
            prediction_coarsened,
            start_timestep,
            start_sample,
            batch_times_coarsened,
        )

    def coarsen_batch(
        self,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        start_timestep: int,
        batch_times: xr.DataArray,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, xr.DataArray,]:
        target_coarsened = self._coarsen_tensor_dict(target)
        prediction_coarsened = self._coarsen_tensor_dict(prediction)
        start_timestep = start_timestep // self._coarsen_factor
        batch_times_coarsened = batch_times.coarsen(
            {TIME_DIM_NAME: self._coarsen_factor}
        ).mean()
        return (
            target_coarsened,
            prediction_coarsened,
            start_timestep,
            batch_times_coarsened,
        )

    def _coarsen_tensor_dict(
        self, tensor_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Coarsen each tensor along a given axis by a given factor."""
        coarsened_tensor_dict = {}
        for name, tensor in tensor_dict.items():
            coarsened_tensor_dict[name] = tensor.unfold(
                dimension=TIME_DIM, size=self._coarsen_factor, step=self._coarsen_factor
            ).mean(dim=-1)
        return coarsened_tensor_dict

    def flush(self):
        self._data_writer.flush()


def tensor_dict_time_select(tensor_dict: Dict[str, torch.Tensor], time_slice: slice):
    return {name: tensor[:, time_slice] for name, tensor in tensor_dict.items()}
