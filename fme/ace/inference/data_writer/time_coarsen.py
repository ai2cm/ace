import dataclasses
from typing import Protocol

import torch
import xarray as xr

TIME_DIM_NAME = "time"
TIME_DIM = 1  # sample, time, lat, lon


class _PairedDataWriter(Protocol):
    def append_batch(
        self,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ):
        pass

    def flush(self):
        pass


class _DataWriter(Protocol):
    def append_batch(
        self,
        data: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
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

    def build_paired(self, data_writer: _PairedDataWriter) -> "PairedTimeCoarsen":
        return PairedTimeCoarsen(
            data_writer=data_writer,
            coarsen_factor=self.coarsen_factor,
        )

    def build(self, data_writer: _DataWriter) -> "TimeCoarsen":
        return TimeCoarsen(
            data_writer=data_writer,
            coarsen_factor=self.coarsen_factor,
        )

    def n_coarsened_timesteps(self, n_timesteps: int) -> int:
        """Assumes initial condition is NOT in n_timesteps."""
        return (n_timesteps) // self.coarsen_factor


class PairedTimeCoarsen:
    """Wraps a data writer and coarsens its arguments in time before passing them on."""

    def __init__(
        self,
        data_writer: _PairedDataWriter,
        coarsen_factor: int,
    ):
        self._data_writer: _PairedDataWriter = data_writer
        self._coarsen_factor: int = coarsen_factor

    def append_batch(
        self,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ):
        (target_coarsened, start_timestep, batch_times_coarsened) = coarsen_batch(
            target, start_timestep, batch_time, self._coarsen_factor
        )
        (prediction_coarsened, _, _) = coarsen_batch(
            prediction, start_timestep, batch_time, self._coarsen_factor
        )
        self._data_writer.append_batch(
            target_coarsened,
            prediction_coarsened,
            start_timestep,
            batch_times_coarsened,
        )

    def flush(self):
        self._data_writer.flush()


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
        data: dict[str, torch.Tensor],
        start_timestep: int,
        batch_time: xr.DataArray,
    ):
        (data_coarsened, start_timestep, batch_times_coarsened) = coarsen_batch(
            data, start_timestep, batch_time, self._coarsen_factor
        )
        self._data_writer.append_batch(
            data_coarsened,
            start_timestep,
            batch_times_coarsened,
        )

    def flush(self):
        self._data_writer.flush()


def coarsen_batch(
    data: dict[str, torch.Tensor],
    start_timestep: int,
    batch_time: xr.DataArray,
    coarsen_factor: int,
) -> tuple[dict[str, torch.Tensor], int, xr.DataArray]:
    data_coarsened = _coarsen_tensor_dict(data, coarsen_factor)
    start_timestep = start_timestep // coarsen_factor
    batch_time_coarsened = batch_time.coarsen({TIME_DIM_NAME: coarsen_factor}).mean()
    return data_coarsened, start_timestep, batch_time_coarsened


def _coarsen_tensor_dict(
    tensor_dict: dict[str, torch.Tensor], coarsen_factor: int
) -> dict[str, torch.Tensor]:
    """Coarsen each tensor along a given axis by a given factor."""
    coarsened_tensor_dict = {}
    for name, tensor in tensor_dict.items():
        coarsened_tensor_dict[name] = tensor.unfold(
            dimension=TIME_DIM, size=coarsen_factor, step=coarsen_factor
        ).mean(dim=-1)
    return coarsened_tensor_dict
