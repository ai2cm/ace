import dataclasses
from collections.abc import Sequence

import xarray as xr

from fme.core.dataset.concat import ConcatDatasetConfig
from fme.core.dataset.config import DatasetConfigABC
from fme.core.dataset.dataset import DatasetABC, DatasetItem
from fme.core.dataset.merged import MergeDatasetConfig
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntMilestone, IntSchedule
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.typing_ import TensorDict


@dataclasses.dataclass
class TimeCoarsenDatasetConfig(DatasetConfigABC):
    """
    Coarsen each sample of a wrapped dataset in time by an integer factor.

    Every ``factor`` consecutive timesteps of the wrapped dataset form one
    coarse timestep. Snapshot variables take their value at the last timestep
    of each window and window variables are averaged over the window; the
    coarse time coordinate is the last time of each window. These are the
    conventions of ``scripts/data_process/time_coarsen.py``, so a sample
    loaded through this wrapper whose windows end at times present in a store
    coarsened offline by that script is identical to the offline store's
    sample. Unlike the offline store, every timestep of the wrapped dataset
    starts a sample, so the wrapper yields ``factor`` times as many samples
    per epoch, each of which begins at a different phase of the coarse step.

    The coarse timestep is ``factor`` times the wrapped dataset's timestep,
    which must have been inferred (``infer_timestep: true``). Inference is not
    supported through this wrapper; use a store coarsened offline for inline
    inference and evaluation.

    Parameters:
        dataset: Configuration of the dataset to coarsen.
        factor: Number of consecutive timesteps of the wrapped dataset per
            coarse timestep.
        window_names: Variables averaged over each window (fluxes and other
            interval means). Any other requested variable is a snapshot.
        snapshot_names: Optional list of snapshot variables. If given, every
            requested variable must appear in it or in ``window_names``, which
            guards against an interval mean being silently subsampled.
    """

    dataset: ConcatDatasetConfig | MergeDatasetConfig | XarrayDataConfig
    factor: int
    window_names: Sequence[str] = ()
    snapshot_names: Sequence[str] | None = None

    def __post_init__(self):
        if self.factor < 1:
            raise ValueError(f"factor must be a positive integer, got {self.factor}")
        if self.snapshot_names is not None:
            overlap = set(self.snapshot_names) & set(self.window_names)
            if overlap:
                raise ValueError(
                    "Variables cannot be both snapshot and window variables: "
                    f"{sorted(overlap)}"
                )

    @property
    def zarr_engine_used(self) -> bool:
        return self.dataset.zarr_engine_used

    @property
    def available_labels(self) -> set[str] | None:
        return self.dataset.available_labels

    def _validate_names(self, names: Sequence[str]):
        if self.snapshot_names is None:
            return
        known = set(self.snapshot_names) | set(self.window_names)
        unknown = sorted(set(names) - known)
        if unknown:
            raise ValueError(
                "Requested variables are in neither snapshot_names nor "
                f"window_names: {unknown}"
            )

    def build(
        self,
        names: Sequence[str],
        n_timesteps: IntSchedule,
        allow_missing_variables: bool = False,
    ) -> tuple[DatasetABC, DatasetProperties]:
        self._validate_names(names)
        fine_n_timesteps = IntSchedule(
            start_value=n_timesteps.start_value * self.factor,
            milestones=[
                IntMilestone(epoch=m.epoch, value=m.value * self.factor)
                for m in n_timesteps.milestones
            ],
        )
        dataset, _ = self.dataset.build(
            names, fine_n_timesteps, allow_missing_variables=allow_missing_variables
        )
        coarsened = TimeCoarsenDataset(dataset, self.factor, self.window_names)
        return coarsened, coarsened.properties


class TimeCoarsenDataset(DatasetABC):
    """
    Wraps a dataset and coarsens each sample in time by an integer factor.

    Each window of ``factor`` consecutive timesteps of the wrapped dataset
    becomes one coarse timestep: window variables are averaged over the
    window, every other variable takes its value at the last timestep of the
    window, and the coarse time is the last time of the window. Samples of
    the wrapped dataset must therefore have a multiple of ``factor``
    timesteps. Note that dataset samples do not have a batch dimension.
    """

    def __init__(self, dataset: DatasetABC, factor: int, window_names: Sequence[str]):
        if factor < 1:
            raise ValueError(f"factor must be a positive integer, got {factor}")
        self._dataset = dataset
        self._factor = factor
        self._window_names = set(window_names)
        fine_timestep = dataset.properties.timestep
        if fine_timestep is None:
            raise ValueError(
                "TimeCoarsenDataset requires the wrapped dataset to have an "
                "inferred timestep (XarrayDataConfig.infer_timestep must be True)."
            )
        self._fine_timestep = fine_timestep
        self._timestep = fine_timestep * factor

    def _coarsen_data(self, data: TensorDict, n_fine: int) -> TensorDict:
        if n_fine % self._factor != 0:
            raise ValueError(
                f"Sample has {n_fine} timesteps, which is not a multiple of the "
                f"coarsening factor {self._factor}."
            )
        n_coarse = n_fine // self._factor
        coarsened: TensorDict = {}
        for name, tensor in data.items():
            if tensor.shape[0] != n_fine:
                raise ValueError(
                    f"Variable {name} has {tensor.shape[0]} timesteps but the "
                    f"sample's time coordinate has {n_fine}."
                )
            if name in self._window_names:
                coarsened[name] = tensor.reshape(
                    n_coarse, self._factor, *tensor.shape[1:]
                ).mean(dim=1)
            else:
                # the snapshot is taken at the end of each window, as window
                # variables are means over the interval leading up to it
                coarsened[name] = tensor[self._factor - 1 :: self._factor]
        return coarsened

    def _coarsen_time(self, time: xr.DataArray) -> xr.DataArray:
        return time.isel(time=slice(self._factor - 1, None, self._factor))

    def _coarsen_item(self, item: DatasetItem) -> DatasetItem:
        data, time, labels, epoch, missing_names = item
        n_fine = time.shape[0]
        return (
            self._coarsen_data(data, n_fine),
            self._coarsen_time(time),
            labels,
            epoch,
            missing_names,
        )

    def __getitem__(self, idx: int) -> DatasetItem:
        return self._coarsen_item(self._dataset[idx])

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        """The coarse initial condition of each sample is the last time of
        the sample's first window.
        """
        offset = (self._factor - 1) * self._fine_timestep
        return self._dataset.sample_start_times + offset

    @property
    def all_times(self) -> xr.CFTimeIndex:
        raise NotImplementedError("TimeCoarsenDataset does not support inference.")

    @property
    def sample_n_times(self) -> int:
        n_fine = self._dataset.sample_n_times
        if n_fine % self._factor != 0:
            raise ValueError(
                f"Wrapped dataset samples have {n_fine} timesteps, which is not "
                f"a multiple of the coarsening factor {self._factor}."
            )
        return n_fine // self._factor

    def get_sample_by_time_slice(self, time_slice: slice) -> DatasetItem:
        raise NotImplementedError(
            "TimeCoarsenDataset does not support getting samples by time slice."
        )

    @property
    def properties(self) -> DatasetProperties:
        properties = self._dataset.properties.copy()
        properties.timestep = self._timestep
        return properties

    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        raise ValueError("TimeCoarsenDataset does not support inference.")

    def enable_shared_memory(self):
        self._dataset.enable_shared_memory()

    def set_global_epoch_tensor(self, tensor):
        self._dataset.set_global_epoch_tensor(tensor)

    def set_epoch(self, epoch: int):
        self._dataset.set_epoch(epoch)
