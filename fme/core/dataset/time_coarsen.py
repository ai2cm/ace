import dataclasses
from typing import Literal, Protocol

import torch
import torch.nn.functional as F
import xarray as xr

from fme.core.typing_ import TensorDict


@dataclasses.dataclass
class TimeCoarsenConfig:
    factor: int
    snapshot_names: list[str]
    window_names: list[str]
    window_coarsen_type: Literal["mean"] = "mean"


class Dataset(Protocol):
    """
    Abstract base class for all datasets.

    Meant mainly for internal use in `fme.core.dataset`, to make it clear what methods
    must be implemented to wrap datasets in other dataset types.
    """

    def __len__(self):
        pass

    def __getitem__(self, idx: int) -> tuple[TensorDict, xr.DataArray]:
        pass

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        pass

    @property
    def sample_n_times(self) -> int:
        pass


class TimeCoarsenDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset and coarsens each sample in time before passing them on.

    Note that dataset samples do not have a batch dimension.
    """

    def __init__(self, dataset: Dataset, config: TimeCoarsenConfig):
        self.dataset = dataset
        self._config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[TensorDict, xr.DataArray]:
        raw_data, raw_time = self.dataset[idx]
        coarsened_data: dict[str, torch.Tensor] = {}
        n_pooled_timesteps = (
            raw_time.shape[0] // self._config.factor
        ) * self._config.factor
        for name in raw_data.keys():
            n_pooled_timesteps = (
                raw_data[name].shape[0] // self._config.factor
            ) * self._config.factor
            if name in self._config.window_names:
                if self._config.window_coarsen_type == "mean":
                    coarsened_data[name] = F.avg_pool1d(
                        raw_data[name][:n_pooled_timesteps].T,
                        kernel_size=self._config.factor,
                        stride=self._config.factor,
                        ceil_mode=False,
                    ).T
                else:
                    raise ValueError(
                        "Unknown window coarsening type: "
                        f"{self._config.window_coarsen_type}"
                    )
            elif name in self._config.snapshot_names:
                # must take the timestep from the _end_ of a window
                # as windowed data is defined as leading up to the snapshot
                # for that timestep
                coarsened_data[name] = raw_data[name][
                    self._config.factor - 1 : n_pooled_timesteps : self._config.factor
                ]
            else:
                raise ValueError(
                    f"Variable {name} is not a window or snapshot variable, "
                    f"window variables are {self._config.window_names}, "
                    f"snapshot variables are {self._config.snapshot_names}"
                )
        return coarsened_data, raw_time[: n_pooled_timesteps : self._config.factor]

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        return self.dataset.sample_start_times

    @property
    def sample_n_times(self) -> int:
        return self.dataset.sample_n_times // self._config.factor
