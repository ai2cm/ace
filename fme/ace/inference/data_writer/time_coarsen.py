import dataclasses
from typing import Literal, Protocol

import torch
import xarray as xr

TIME_DIM_NAME = "time"
TIME_DIM = 1  # sample, time, lat, lon


class _PairedDataWriter(Protocol):
    def append_batch(
        self,
        target: dict[str, torch.Tensor],
        prediction: dict[str, torch.Tensor],
        batch_time: xr.DataArray,
    ):
        pass

    def flush(self):
        pass

    def finalize(self):
        pass


class _DataWriter(Protocol):
    def append_batch(
        self,
        data: dict[str, torch.Tensor],
        batch_time: xr.DataArray,
    ):
        pass

    def flush(self):
        pass

    def finalize(self):
        pass


@dataclasses.dataclass
class TimeCoarsenConfig:
    """
    Config for inference data time coarsening.

    Args:
        coarsen_factor: Factor by which to coarsen in time, an integer 1 or greater. The
            resulting time labels will be coarsened to the mean of the original labels.
        method: Method to use for coarsening, currently only "block_mean" is supported.
    """

    def __post_init__(self):
        if self.coarsen_factor < 1:
            raise ValueError(
                f"coarsen_factor must be 1 or greater, got {self.coarsen_factor}"
            )

    coarsen_factor: int
    method: Literal["block_mean"] = "block_mean"

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

    def validate(self, forward_steps_in_memory: int, n_forward_steps: int):
        if forward_steps_in_memory % self.coarsen_factor != 0:
            raise ValueError(
                "forward_steps_in_memory must be divisible by "
                f"time_coarsen.coarsen_factor. Got {forward_steps_in_memory} "
                f"and {self.coarsen_factor}."
            )
        if n_forward_steps % self.coarsen_factor != 0:
            raise ValueError(
                "n_forward_steps must be divisible by "
                f"time_coarsen.coarsen_factor. Got {n_forward_steps} "
                f"and {self.coarsen_factor}."
            )


@dataclasses.dataclass
class MonthlyCoarsenConfig:
    """
    Config for inference data monthly coarsening.

    Args:
        method: Method to use for coarsening, currently only "monthly_mean" is
            supported.
    """

    method: Literal["monthly_mean"] = "monthly_mean"

    def validate(self, forward_steps_in_memory: int, n_forward_steps: int):
        # monthly coarsening works with any combination of steps
        pass


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
        batch_time: xr.DataArray,
    ):
        (target_coarsened, batch_times_coarsened) = coarsen_batch(
            target, batch_time, self._coarsen_factor
        )
        (prediction_coarsened, _) = coarsen_batch(
            prediction, batch_time, self._coarsen_factor
        )
        self._data_writer.append_batch(
            target_coarsened,
            prediction_coarsened,
            batch_times_coarsened,
        )

    def flush(self):
        self._data_writer.flush()

    def finalize(self):
        self._data_writer.finalize()


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
        batch_time: xr.DataArray,
    ):
        (data_coarsened, batch_times_coarsened) = coarsen_batch(
            data, batch_time, self._coarsen_factor
        )
        self._data_writer.append_batch(
            data_coarsened,
            batch_times_coarsened,
        )

    def flush(self):
        self._data_writer.flush()

    def finalize(self):
        self._data_writer.finalize()


def coarsen_batch(
    data: dict[str, torch.Tensor],
    batch_time: xr.DataArray,
    coarsen_factor: int,
) -> tuple[dict[str, torch.Tensor], xr.DataArray]:
    data_coarsened = _coarsen_tensor_dict(data, coarsen_factor)
    batch_time_coarsened = batch_time.coarsen({TIME_DIM_NAME: coarsen_factor}).mean()
    return data_coarsened, batch_time_coarsened


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
