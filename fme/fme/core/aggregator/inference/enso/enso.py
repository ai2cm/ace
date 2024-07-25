import datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import cftime
import matplotlib.pyplot as plt
import torch
import xarray as xr

from fme.core.aggregator.plotting import get_cmap_limits, plot_imshow, plot_paneled_data
from fme.core.data_loading.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.metrics import root_mean_squared_error
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import WandB

from .index import INDEX_CALENDAR, NINO34_INDEX

OVERLAP_THRESHOLD = 0.9


def index_data_array(
    index_data: List[Tuple[Tuple[int, int, int], float]], calendar: str
) -> xr.DataArray:
    """Convert a list of (time, index) tuples to an xarray DataArray.

    Args:
        index_data: List of (time, index) tuples.

    Returns:
        ENSO index data as an xarray DataArray.
    """
    timestamps, index_values = zip(*index_data)
    time_coord = xr.DataArray(
        [cftime.datetime(*timestamp, calendar=calendar) for timestamp in timestamps],
        dims=["time"],
    )
    return xr.DataArray(list(index_values), coords={"time": time_coord}, dims=["time"])


class EnsoCoefficientEvaluatorAggregator:
    _image_captions = {
        "target_gen_coefficient_maps": (
            "{name} target (L) and generated (R) coefficient "
            "with 1940-2020 CMIP6 AMIP Nino 3.4 index [{units} / K]; "
            "metric is not meaningful if a different SST dataset or "
            "interactive ocean is used;"
        ),
        "coefficient_difference_map": (
            "{name} difference in coefficient "
            "with 1940-2020 CMIP6 AMIP Nino 3.4 (generated - target) [{units} / K];"
            "metric is not meaningful if a different SST dataset or "
            "interactive ocean is used;"
        ),
    }

    _enso_index: xr.DataArray = index_data_array(NINO34_INDEX, INDEX_CALENDAR)

    """Compute coefficients of variables regressed against a pre-computed
    scalar ENSO index (i.e., a global-mean time series of ENSO index values).

    We compute the spatially-varying regression coefficients (a map) of each
    predicted variable against the ENSO index, using a simplified covariance-
    over-variance formula for the coefficients that assumes that the index has
    zero mean. For variable i, the coefficients are given by:

        coefficient_i = sum_t(data_it * index_t) / sum_t(index_t ** 2)

    Because the index has zero-mean, this is equivalent to the covariance of the
    data with the index divided by the variance of the index. The computation is
    implemented via running sums over predicted timesteps t, which has the
    advantage of not depending on the number of timesteps in the data, or whether
    data are time-aggregated.

    Args:
        initial_times: Initial times for each sample.
        n_forward_timesteps: Number of timesteps for each sample.
        timestep: Timestep duration.
        area_weights: Area weights for spatial averaging.
        metadata: Metadata for the variables in the data.
    """

    def __init__(
        self,
        initial_times: xr.DataArray,
        n_forward_timesteps: int,
        timestep: datetime.timedelta,
        area_weights: torch.Tensor,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ):
        self._sample_index_series: List[
            Optional[xr.DataArray]
        ] = get_sample_index_series(
            self.enso_index, initial_times, n_forward_timesteps, timestep
        )
        self._area_weights: torch.Tensor = area_weights
        if metadata is not None:
            self._metadata: Mapping[str, VariableMetadata] = metadata
        else:
            self._metadata = {}
        n_samples = len(self._sample_index_series)
        self._target_covariances: List[TensorDict] = [{} for _ in range(n_samples)]
        self._gen_covariances: List[TensorDict] = [{} for _ in range(n_samples)]
        self._index_variance: List[torch.Tensor] = [
            torch.tensor(0.0, dtype=torch.float32, device=get_device())
            for _ in range(n_samples)
        ]

    @property
    def enso_index(self) -> xr.DataArray:
        return self._enso_index

    @torch.no_grad()
    def record_batch(
        self,
        time: xr.DataArray,
        target_data: TensorMapping,
        gen_data: TensorMapping,
    ):
        """Record running sums of the enso index variance, and of the
        covariance of the target and generated data with the ENSO index (sum
        of squares is used as a proxy for variance/covariance).

        We need to track sums for each sample since the index will be different
        for each time period.
        """
        assert time.sizes["sample"] == len(
            self._sample_index_series
        ), "number of index series must match number of samples"
        for i_sample, sample_index_series in enumerate(self._sample_index_series):
            if sample_index_series is not None:
                sample_index_series_reindexed = sample_index_series.reindex(
                    time=time.isel(sample=i_sample), method="nearest"
                ).values
                sample_index_series_reindexed = torch.tensor(
                    sample_index_series_reindexed,
                    device=get_device(),
                    dtype=torch.float32,
                )
                self._index_variance[i_sample] += (
                    sample_index_series_reindexed**2
                ).sum()
                for name, data in target_data.items():
                    if name not in self._target_covariances[i_sample]:
                        self._target_covariances[i_sample][
                            name
                        ] = data_index_covariance(
                            data[i_sample, :], sample_index_series_reindexed
                        )
                    else:
                        self._target_covariances[i_sample][
                            name
                        ] += data_index_covariance(
                            data[i_sample, :], sample_index_series_reindexed
                        )
                for name, data in gen_data.items():
                    if name not in self._gen_covariances[i_sample]:
                        self._gen_covariances[i_sample][name] = data_index_covariance(
                            data[i_sample, :], sample_index_series_reindexed
                        )
                    else:
                        self._gen_covariances[i_sample][name] += data_index_covariance(
                            data[i_sample, :], sample_index_series_reindexed
                        )

    def _compute_coefficients(
        self, which: Literal["target", "gen"]
    ) -> List[TensorDict]:
        """Compute the coefficients of the target or generated data regressed
        against the ENSO index for each spatial grid cell for each sample.
        """
        if which == "target":
            covariances = self._target_covariances
        elif which == "gen":
            covariances = self._gen_covariances
        coefficients: List[TensorDict] = [{} for _ in range(len(covariances))]
        for i_sample in range(len(self._index_variance)):
            if self._sample_index_series[i_sample] is not None:
                for name, covariance in covariances[i_sample].items():
                    coefficients[i_sample][name] = (
                        covariance / self._index_variance[i_sample]
                    ).to(device=get_device())
        return coefficients

    def _get_coefficients(self) -> Tuple[Optional[TensorDict], Optional[TensorDict]]:
        dist = Distributed.get_instance()
        target_coefficients = self._compute_coefficients("target")
        gen_coefficients = self._compute_coefficients("gen")
        # average coefficients across samples
        target_coefficients_all, gen_coefficients_all = {}, {}
        target_names = set(
            [
                name
                for target_coefficient in target_coefficients
                for name in target_coefficient.keys()
            ]
        )
        for name in target_names:
            target_coefficients_all[name] = (
                torch.stack(
                    [
                        target_coefficient[name]
                        for target_coefficient in target_coefficients
                        if name in target_coefficient
                    ],
                    dim=0,
                )
                .mean(dim=0)
                .to(device=get_device())
            )
        gen_names = set(
            [
                name
                for gen_coefficient in gen_coefficients
                for name in gen_coefficient.keys()
            ]
        )
        for name in gen_names:
            gen_coefficients_all[name] = (
                torch.stack(
                    [
                        gen_coefficient[name]
                        for gen_coefficient in gen_coefficients
                        if name in gen_coefficient
                    ],
                    dim=0,
                )
                .mean(dim=0)
                .to(device=get_device())
            )
        # average coefficients across processes
        if target_coefficients_all:
            reduced_target_coefficients = reduce_data(dist, target_coefficients_all)
        else:
            reduced_target_coefficients = None
        if gen_coefficients_all:
            reduced_gen_coefficients = reduce_data(dist, gen_coefficients_all)
        else:
            reduced_gen_coefficients = None
        return reduced_target_coefficients, reduced_gen_coefficients

    @torch.no_grad()
    def get_logs(self, label: str) -> Dict[str, Any]:
        target_coefficients, gen_coefficients = self._get_coefficients()
        if target_coefficients is None or gen_coefficients is None:
            return {}  # only the root process returns logs
        wandb = WandB.get_instance()
        images, metrics = {}, {}
        for name in gen_coefficients.keys():
            if name in self._metadata:
                caption_name = self._metadata[name].long_name
                caption_units = self._metadata[name].units
            else:
                caption_name = name
                caption_units = "unknown units"
            caption = self._image_captions["target_gen_coefficient_maps"].format(
                name=caption_name, units=caption_units
            )
            panels = [
                [
                    target_coefficients[name].cpu().numpy(),
                    gen_coefficients[name].cpu().numpy(),
                ]
            ]
            coefficient_map = plot_paneled_data(
                data=panels, diverging=True, caption=caption
            )
            images.update({f"coefficient_maps/{name}": coefficient_map})
            rmse = float(
                root_mean_squared_error(
                    predicted=gen_coefficients[name],
                    truth=target_coefficients[name],
                    weights=self._area_weights,
                )
                .cpu()
                .numpy()
            )
            metrics.update({f"rmse/{name}": rmse})
            diff = gen_coefficients[name] - target_coefficients[name]
            caption = self._image_captions["coefficient_difference_map"].format(
                name=caption_name, units=caption_units
            )
            vmin, vmax = get_cmap_limits(diff.cpu().numpy(), diverging=True)
            caption += (
                f" vmin={vmin:.4g}, vmax={vmax:.4g}; global-mean RMSE={rmse:.4g}."
            )
            fig = plot_imshow(diff.cpu().numpy(), vmin, vmax, cmap="RdBu_r")
            coefficient_difference_map = wandb.Image(fig, caption=caption)
            plt.close(fig)
            images.update(
                {
                    f"coefficient_difference_map/{name}": coefficient_difference_map,
                }
            )
        logs: Dict[str, Any] = {}
        if len(label) > 0:
            label = label + "/"
        logs.update({f"{label}{name}": images[name] for name in images.keys()})
        logs.update({f"{label}{name}": metrics[name] for name in metrics.keys()})
        return logs


def get_sample_index_series(
    index_data: xr.DataArray,
    initial_times: xr.DataArray,
    n_forward_timesteps: int,
    timestep: datetime.timedelta,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> List[Optional[xr.DataArray]]:
    """Get a zero-mean index series for each sample, based on the time that
    sample will run for.

    Args:
        index_data: ENSO index data with a time coordinate.
        initial_times: Initial times for each sample.
        n_forward_timesteps: Number of forward timesteps for each sample.
        timestep: Timestep duration.
        overlap_threshold: Required overlap of reference index with inference period.

    Returns:
        List of zero-mean index series for each sample, or None if the sample does
        not overlap sufficiently with the reference index.
    """
    data_calendar = initial_times.dt.calendar
    index_calendar = index_data.time.dt.calendar
    if data_calendar != index_calendar:
        index_data = index_data.convert_calendar(
            calendar=data_calendar, dim="time", use_cftime=True
        )
    sample_index_series: List[Optional[xr.DataArray]] = []
    for initial_time in initial_times:
        duration = n_forward_timesteps * timestep
        end_time = initial_time + duration
        # select index data that overlaps with the inference period, plus a
        # half-timestep buffer since we will later reindex with nearest neighbor
        index_timestep_seconds = (
            index_data.time[1].item() - index_data.time[0].item()
        ).total_seconds()
        half_index_timestep = datetime.timedelta(seconds=index_timestep_seconds / 2)
        index_series = index_data.sel(
            time=slice(
                initial_time - half_index_timestep,
                end_time + half_index_timestep,
            )
        )
        if index_series.sizes["time"] == 0:
            sample_index_series.append(None)
        else:
            index_series_duration = (
                index_series.time[-1].item() - index_series.time[0].item()
            )
            if index_series_duration > overlap_threshold * duration:
                index_series = index_series - index_series.mean()
                sample_index_series.append(index_series)
            else:
                sample_index_series.append(None)
    return sample_index_series


def data_index_covariance(
    data: torch.Tensor, index_values: torch.Tensor, index_dim: int = 0
) -> torch.Tensor:
    """Compute the product of n-dimensional data and a 1-d index,
    where one of the data dimensions aligns with the index. Covariance
    here just means sum of the products of the data and index values.

    Args:
        data: Data tensor with shape that includes index dimension.
        index_values: Index values with shape (n_index,)
        index_dim: Dimension of the index in the data tensor.

    Returns:
        Covariance of data with index
    """
    n_index = index_values.size(0)
    assert data.size(index_dim) == n_index, "aligned dimension must match index length"
    view_dims = [1 if i != index_dim else n_index for i in range(data.dim())]
    index_values_broadcast = index_values.view(*view_dims)
    return (data * index_values_broadcast).sum(dim=index_dim)


def reduce_data(
    dist: Distributed, rank_tensor_dict: TensorDict
) -> Optional[TensorDict]:
    """Reduce tensor dicts across distributed processes by taking the mean.

    Args:
        dist: Distributed instance.
        rank_tensor_dict: Tensor dict to reduce.

    Returns:
        Reduced tensor dict.
    """
    if dist.is_distributed():
        # sort for determinism
        names = sorted(list(rank_tensor_dict.keys()))
        rank_tensor = torch.stack([rank_tensor_dict[name] for name in names], dim=0)
        reduced_tensor = dist.reduce_mean(rank_tensor)
        gathered_tensor_dict = {name: reduced_tensor[i] for i, name in enumerate(names)}
    else:
        gathered_tensor_dict = rank_tensor_dict
    if dist.is_root():
        return gathered_tensor_dict
    else:
        return None
