import dataclasses
from collections.abc import Mapping
from typing import Any

import cftime
import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Image

from ..plotting import plot_paneled_data
from .build_context import MetricBuildContext, MetricNotSupportedError, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator

SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
# A fixed reference epoch is used (rather than each sample's start time) so the
# running sums stay consistent across batches and distributed ranks without
# communication. The regression slope is shift-invariant, so the epoch does not
# affect the result; it is chosen near the climate-data era only to keep the
# magnitude of the time regressor small, which improves the floating-point
# conditioning of the (float64) running sums.
_REFERENCE_UNITS = "seconds since 2000-01-01 00:00:00"


def _years_since_reference(time: xr.DataArray) -> np.ndarray:
    """Convert a ``(sample, time)`` datetime array to floating-point years
    since a fixed reference epoch.
    """
    calendar = time.dt.calendar
    seconds = cftime.date2num(
        time.values.ravel(), units=_REFERENCE_UNITS, calendar=calendar
    ).reshape(time.shape)
    return seconds / SECONDS_PER_YEAR


class TrendEvaluatorAggregator:
    """Per-grid-cell linear trend (least-squares slope vs. time) maps.

    The trend of each variable at each grid cell is the ordinary least-squares
    slope of its value against time. It is computed by accumulating the
    regression sufficient statistics -- the running sums of ``t``, ``t**2``,
    ``y``, ``t*y`` and the observation count ``n`` -- in a single streaming
    pass over the inference batches. Because only these sums are retained, the
    memory cost is independent of the number of timesteps, so trends can be
    computed for runs whose full time series does not fit in memory.

    For grid cell with values ``y`` at times ``t`` (in years), the slope is

        slope = (n * sum(t*y) - sum(t) * sum(y)) / (n * sum(t**2) - sum(t)**2)

    Both the predicted and target trend maps are produced, along with a bias
    map (generated - target) and the area-weighted RMSE between them. Slope
    units are ``<variable units> / year``.
    """

    _image_captions = {
        "trend_maps": (
            "{name} linear trend; (left) target and (right) generated "
            "[{units} / year]"
        ),
        "trend_difference_map": (
            "{name} linear trend difference (generated - target) [{units} / year]"
        ),
    }

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        horizontal_dims: list[str],
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        """
        Args:
            gridded_operations: Computes gridded (area-weighted) operations.
            horizontal_dims: Names of the horizontal dimensions, used for the
                output dataset.
            variable_metadata: Mapping of variable names to metadata used in
                image captions and dataset attributes.
        """
        self._ops = gridded_operations
        self._horizontal_dims = horizontal_dims
        self._variable_metadata = variable_metadata or {}
        device = get_device()
        # time regressor statistics, shared across variables and grid cells
        self._n = torch.zeros((), dtype=torch.float64, device=device)
        self._sum_t = torch.zeros((), dtype=torch.float64, device=device)
        self._sum_tt = torch.zeros((), dtype=torch.float64, device=device)
        # per-variable [n_lat, n_lon] running sums of y and t*y
        self._target_sum_y: TensorDict = {}
        self._target_sum_ty: TensorDict = {}
        self._gen_sum_y: TensorDict = {}
        self._gen_sum_ty: TensorDict = {}

    @staticmethod
    def _add_running_sums(
        sum_y: TensorDict,
        sum_ty: TensorDict,
        tensor_data: TensorMapping,
        t: torch.Tensor,
    ) -> None:
        """Accumulate sum(y) and sum(t*y) over the (sample, time) dimensions.

        Args:
            sum_y: Running sum of y, mutated in place.
            sum_ty: Running sum of t*y, mutated in place.
            tensor_data: Mapping of variable name to a (sample, time, lat, lon)
                tensor, already sliced to the timesteps being recorded.
            t: Time in years with shape (sample, time), matching ``tensor_data``.
        """
        t_broadcast = t[:, :, None, None]
        for name, tensor in tensor_data.items():
            y = tensor.to(torch.float64)
            contrib_y = y.sum(dim=(0, 1))
            contrib_ty = (t_broadcast * y).sum(dim=(0, 1))
            if name in sum_y:
                sum_y[name] = sum_y[name] + contrib_y
                sum_ty[name] = sum_ty[name] + contrib_ty
            else:
                sum_y[name] = contrib_y
                sum_ty[name] = contrib_ty

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData) -> None:
        # Mirror TimeMeanAggregator: when the initial condition is the first
        # timestep of the first batch, skip it so it is not double-counted.
        time_slice = slice(1, None) if data.i_time_start == 0 else slice(None)
        years = _years_since_reference(data.time)[:, time_slice]
        t = torch.tensor(years, dtype=torch.float64, device=get_device())
        if t.shape[1] == 0:
            return
        self._n = self._n + t.numel()
        self._sum_t = self._sum_t + t.sum()
        self._sum_tt = self._sum_tt + (t * t).sum()
        gen = {name: tensor[:, time_slice] for name, tensor in data.prediction.items()}
        self._add_running_sums(self._gen_sum_y, self._gen_sum_ty, gen, t)
        target = {name: tensor[:, time_slice] for name, tensor in data.target.items()}
        self._add_running_sums(self._target_sum_y, self._target_sum_ty, target, t)

    def _compute_trends(
        self, sum_y: TensorDict, sum_ty: TensorDict
    ) -> TensorDict | None:
        """Reduce sufficient statistics across ranks and finalize the slopes.

        Returns ``None`` on non-root ranks.
        """
        dist = Distributed.get_instance()
        # reduce additive sufficient statistics across data-parallel ranks;
        # clone first so repeated get_logs/get_dataset calls stay correct.
        n = dist.reduce_sum(self._n.clone())
        sum_t = dist.reduce_sum(self._sum_t.clone())
        sum_tt = dist.reduce_sum(self._sum_tt.clone())
        names = sorted(sum_y.keys())
        stacked_y = dist.reduce_sum(torch.stack([sum_y[name] for name in names], dim=0))
        stacked_ty = dist.reduce_sum(
            torch.stack([sum_ty[name] for name in names], dim=0)
        )
        if not dist.is_root():
            return None
        denom = n * sum_tt - sum_t * sum_t
        trends: TensorDict = {}
        for i, name in enumerate(names):
            numerator = n * stacked_ty[i] - sum_t * stacked_y[i]
            trends[name] = numerator / denom
        return trends

    def _get_trends(self) -> tuple[TensorDict | None, TensorDict | None]:
        if not self._gen_sum_y:
            raise ValueError("No data has been recorded yet.")
        target_trends = self._compute_trends(self._target_sum_y, self._target_sum_ty)
        gen_trends = self._compute_trends(self._gen_sum_y, self._gen_sum_ty)
        return target_trends, gen_trends

    def _caption(self, key: str, name: str) -> str:
        if name in self._variable_metadata:
            caption_name = self._variable_metadata[name].display_long_name(name)
            units = self._variable_metadata[name].display_units("unknown units")
        else:
            caption_name, units = name, "unknown units"
        return self._image_captions[key].format(name=caption_name, units=units)

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, float | Image]:
        target_trends, gen_trends = self._get_trends()
        if target_trends is None or gen_trends is None:
            return {}  # only the root process returns logs
        images: dict[str, Image] = {}
        metrics: dict[str, float] = {}
        for name in gen_trends:
            target_map = target_trends[name]
            gen_map = gen_trends[name]
            images[f"trend_maps/{name}"] = plot_paneled_data(
                [[target_map.cpu().numpy(), gen_map.cpu().numpy()]],
                diverging=True,
                caption=self._caption("trend_maps", name),
            )
            difference = gen_map - target_map
            images[f"trend_difference_map/{name}"] = plot_paneled_data(
                [[difference.cpu().numpy()]],
                diverging=True,
                caption=self._caption("trend_difference_map", name),
            )
            metrics[f"rmse/{name}"] = float(
                self._ops.area_weighted_rmse(
                    predicted=gen_map.to(torch.float32),
                    truth=target_map.to(torch.float32),
                    name=name,
                )
                .cpu()
                .numpy()
            )
        logs: dict[str, Any] = {}
        if len(label) > 0:
            label = label + "/"
        logs.update({f"{label}{key}": image for key, image in images.items()})
        logs.update({f"{label}{key}": metric for key, metric in metrics.items()})
        return logs

    def _var_attrs(self, name: str) -> dict[str, str]:
        if name in self._variable_metadata:
            long_name = self._variable_metadata[name].display_long_name(name)
            units = self._variable_metadata[name].display_units("unknown units")
        else:
            long_name, units = name, "unknown units"
        return {
            "long_name": f"{long_name} linear trend",
            "units": f"{units} / year",
        }

    def get_dataset(self) -> xr.Dataset:
        target_trends, gen_trends = self._get_trends()
        if target_trends is None or gen_trends is None:
            return xr.Dataset()
        target_ds = xr.Dataset(
            {
                name: (
                    self._horizontal_dims,
                    value.cpu().numpy(),
                    self._var_attrs(name),
                )
                for name, value in target_trends.items()
            }
        ).expand_dims({"source": ["target"]})
        gen_ds = xr.Dataset(
            {
                name: (
                    self._horizontal_dims,
                    value.cpu().numpy(),
                    self._var_attrs(name),
                )
                for name, value in gen_trends.items()
            }
        ).expand_dims({"source": ["prediction"]})
        return xr.concat([target_ds, gen_ds], dim="source")


@dataclasses.dataclass
class TrendMetricConfig:
    """Configuration for the linear-trend-map inference metric.

    Attributes:
        variables: Variables to compute trends for. If ``None``, all available
            variables are included.
        name: Name used to label the metric's logs and diagnostics.
        enabled: Whether the metric is computed. Disabled by default.
        strict: If True, raise rather than skip when the metric is not
            supported for the current configuration.
    """

    variables: list[str] | None = None
    name: str = "trend"
    enabled: bool = False
    strict: bool = False

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        if ctx.n_timesteps < 2:
            raise MetricNotSupportedError(
                f"trend metric requires at least 2 timesteps, got {ctx.n_timesteps}"
            )
        agg: SubAggregator = TrendEvaluatorAggregator(
            ctx.ops,
            horizontal_dims=list(ctx.horizontal_coordinates.dims),
            variable_metadata=ctx.variable_metadata,
        )
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))
