import dataclasses
from typing import Literal, Optional

from fme.core.distributed import Distributed


@dataclasses.dataclass
class Slice:
    """
    Configuration of a python `slice` built-in.

    Required because `slice` cannot be initialized directly by dacite.
    """

    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop, self.step)


@dataclasses.dataclass
class XarrayDataParams:
    """
    Attributes:
        data_path: Path to the data.
        n_repeats: Number of times to repeat the dataset (in time).
        engine: Backend for xarray.open_dataset. Currently supported options
            are "netcdf4" (the default) and "h5netcdf". Only valid when using
            XarrayDataset.
    """

    data_path: str
    n_repeats: int = 1
    engine: Optional[Literal["netcdf4", "h5netcdf"]] = None


@dataclasses.dataclass
class DataLoaderParams:
    """
    Attributes:
        dataset: Parameters to define the dataset.
        batch_size: Number of samples per batch.
        num_data_workers: Number of parallel workers to use for data loading.
        data_type: Type of data to load.
        n_samples: Number of samples to load, starting at the beginning of the data.
            If None, load all samples.
        window_starts: Slice indicating the set of indices to consider for initial
            conditions of windows of data. Values following the initial condition will
            still come from the full dataset. By default load all initial conditions.
    """

    dataset: XarrayDataParams
    batch_size: int
    num_data_workers: int
    data_type: Literal["xarray", "ensemble_xarray"]
    n_samples: Optional[int] = None
    window_starts: Slice = dataclasses.field(default_factory=Slice)

    def __post_init__(self):
        if self.n_samples is not None and self.batch_size > self.n_samples:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be less than or equal to "
                f"n_samples ({self.n_samples}) or no batches would be produced"
            )
        if self.data_type not in ["xarray", "ensemble_xarray"]:
            if self.engine is not None:
                raise ValueError(
                    f"Got engine={self.engine}, but "
                    f"engine is unused for data_type {self.data_type}. "
                    "Did you mean to use data_type "
                    '"xarray" or "ensemble_xarray"?'
                )
        dist = Distributed.get_instance()
        if self.batch_size % dist.world_size != 0:
            raise ValueError(
                "batch_size must be divisible by the number of parallel "
                f"workers, got {self.batch_size} and {dist.world_size}"
            )

        if self.dataset.n_repeats != 1 and self.data_type == "ensemble_xarray":
            raise ValueError("n_repeats must be 1 when using ensemble_xarray")
