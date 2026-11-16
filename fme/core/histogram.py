import collections
import math
from collections import namedtuple
from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from fme.core.metrics import quantile
from fme.core.typing_ import TensorMapping

EPSILON = 1.0e-6
_OVERFLOW_SCALE = 2.0**-4

_Histogram = namedtuple("_Histogram", ["counts", "bin_edges"])


def _add_trailing_slash(s):
    if len(s) == 0 or s.endswith("/"):
        return s
    else:
        return s + "/"


def _format_percentile_for_metric_key(p: float, reference_percentile: float) -> str:
    r"""
    Format *p* (the percentile passed to tail routines) for wandb keys.

    When *p* matches the configured reference (upper tail), use ``str(reference)``
    so keys match e.g. ``99.9th-percentile``. For complementary tails
    (``p = 100 - reference``), round using the reference's fractional width
    (e.g. ref ``99.9999`` -> ``0.0001``).
    """
    if p == reference_percentile:
        return str(reference_percentile)

    # decimal places in the reference percentile
    s = repr(reference_percentile)
    if "." not in s:
        return str(int(s))
    decimal_places = len(s.split(".")[1])

    rounded = round(p, decimal_places)
    if decimal_places == 0:
        return str(int(rounded))
    return f"{rounded:.{decimal_places}f}"


def trim_zero_bins(
    counts: np.ndarray, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trim bins with zero counts from the edges of the histogram.

    Args:
        counts: 1D array of counts for each bin
        bin_edges: 1D array of bin edges

    Returns:
        counts, bin_edges: trimmed arrays
    """
    mask = counts > 0
    first_nonzero = np.argmax(mask)
    last_nonzero = len(mask) - np.argmax(mask[::-1])
    return (
        counts[first_nonzero:last_nonzero],
        bin_edges[first_nonzero : last_nonzero + 1],
    )


def histogram(
    arr: torch.Tensor, vmin: float, bin_size: float, n_bins: int
) -> torch.Tensor:
    """
    Compute histogram of a tensor. Values outside of the range of the bins
    are counted in the first or last bin.

    Args:
        arr: 1D tensor of finite values
        vmin: minimum value of the histogram
        bin_size: size of each bin
        n_bins: number of bins

    Returns:
        counts: tensor of counts for each bin
    """
    if arr.dtype != torch.float64:
        # float64 is not available on all devices
        arr = arr.to(torch.float32)
    # the range of the bins can be larger than the largest value which can be
    # represented in the dtype of arr, scaling by a power of two avoids
    # overflow without changing the result
    scale = _OVERFLOW_SCALE
    # in-place operations on the new tensor avoid further allocations
    i_bin = arr * scale
    i_bin -= vmin * scale
    i_bin /= bin_size * scale
    i_bin.clamp_(0, n_bins - 1)
    # use bincount for speed
    return torch.bincount(i_bin.int(), minlength=n_bins)


def _finite_values(value: torch.Tensor) -> tuple[torch.Tensor, float, float] | None:
    """
    Get the finite values of a tensor and their range.

    Args:
        value: 1D tensor of values

    Returns:
        The finite values with their minimum and maximum, or None if there
        are no finite values.
    """
    if value.numel() == 0:
        return None
    vmin, vmax = _min_max(value)
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        value = value[torch.isfinite(value)]
        if value.numel() == 0:
            return None
        vmin, vmax = _min_max(value)
    return value, vmin, vmax


def _min_max(value: torch.Tensor) -> tuple[float, float]:
    # the result is non-finite if any value is non-finite
    vmin, vmax = torch.aminmax(value)
    return float(vmin), float(vmax)


class DynamicHistogram:
    """
    Histograms of one or more series of values, which dynamically bin values
    into a fixed number of bins of constant width. A separate histogram is
    kept for each series, and the same bins are used for all series, so
    that the histograms can be compared bin by bin.

    When a new value is added to any series that goes out of range of the
    current bins, bins are doubled in size until that value is within the
    range of the bins.
    """

    def __init__(self, n_series: int = 1, n_bins: int = 300):
        """
        Args:
            n_series: Number of series of values to keep histograms of.
            n_bins: Number of bins to use for the histogram, must be even.
                The "effective" number of bins may be as small as 1/4th this
                number of bins, as there may be bins greater than the max or
                less than the min value with no samples in them, due to the
                dynamic resizing of bins. For any one series it may be smaller
                still, as the bins span the range of the values of all series.
        """
        if n_bins % 2 != 0:
            raise ValueError(
                f"n_bins must be even so that bins can be doubled in size, got {n_bins}"
            )
        self._n_series = n_series
        self._n_bins = n_bins
        self.bin_edges: np.ndarray | None = None
        self.counts = np.zeros((n_series, n_bins), dtype=np.int64)
        self._epsilon: float = EPSILON

    def add(self, values: Sequence[torch.Tensor]):
        """
        Add new values to the histograms. Non-finite values are ignored.

        Args:
            values: 1D tensor of values to add for each series, which may
                differ in length between series.
        """
        if len(values) != self._n_series:
            raise ValueError(
                f"expected values for {self._n_series} series, got {len(values)}"
            )
        finite_values = [_finite_values(value) for value in values]
        ranges = [(vmin, vmax) for _, vmin, vmax in filter(None, finite_values)]
        if len(ranges) == 0:
            return
        # the bins must contain the values of all series
        vmin = min(vmin for vmin, _ in ranges)
        vmax = max(vmax for _, vmax in ranges)
        # pad the range to ensure all values stay within (and not just equal
        # to) the bin edges, and to avoid the case where vmin == vmax
        pad = self._epsilon * max(1.0, abs(vmin), abs(vmax))
        vmin, vmax = vmin - pad, vmax + pad

        if self.bin_edges is None:
            self.bin_edges = np.linspace(vmin, vmax, self._n_bins + 1)
        else:
            while vmin < self.bin_edges[0]:
                self._double_size("left")
            while vmax > self.bin_edges[-1]:
                self._double_size("right")

        for i_series, finite in enumerate(finite_values):
            if finite is not None:
                self.counts[i_series] += (
                    histogram(
                        finite[0],
                        self.bin_edges[0],
                        self.bin_edges[1] - self.bin_edges[0],
                        self._n_bins,
                    )
                    .cpu()
                    .numpy()
                )

    def _double_size(self, direction: Literal["left", "right"]):
        """
        Double the sizes of bins, extending the histograms to the left
        (further negative) or to the right (further positive).
        """
        if self.bin_edges is None:
            raise RuntimeError("Cannot double size of bins without bin edges")
        new_range = 2 * (self.bin_edges[-1] - self.bin_edges[0])
        new_counts = np.zeros_like(self.counts)
        combined_counts = self.counts[:, ::2] + self.counts[:, 1::2]
        if direction == "left":
            start, stop = self.bin_edges[-1] - new_range, self.bin_edges[-1]
            new_counts[:, self._n_bins // 2 :] = combined_counts
        else:
            start, stop = self.bin_edges[0], self.bin_edges[0] + new_range
            new_counts[:, : self._n_bins // 2] = combined_counts
        self.bin_edges = np.linspace(start, stop, self._n_bins + 1)
        self.counts = new_counts

    def get_bin_edges(self) -> np.ndarray:
        """
        Get the bin edges, which are NaN if no finite values have been added.
        """
        if self.bin_edges is None:
            return np.full(self._n_bins + 1, np.nan)
        return self.bin_edges


class DynamicHistogramAggregator:
    def __init__(
        self,
        n_bins: int,
        percentiles: list[float] | None = None,
        nan_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.n_bins = n_bins
        self.percentiles = [99.9999] if percentiles is None else percentiles
        self.histograms: Mapping[str, DynamicHistogram] | None = None
        self.nan_masks = nan_masks

    def set_nan_masks(self, data: TensorMapping):
        if self.nan_masks is None:
            # assume the final two dimensions are the spatial dimensions and the
            # spatial pattern of NaNs is the same for all samples & time, but
            # can vary by variable
            ndim = next(iter(data.values())).ndim
            index = (slice(0, 1),) * (ndim - 2) + (slice(None), slice(None))
            self.nan_masks = {
                k: v[index].isnan() if torch.any(v[index].isnan()) else None
                for k, v in data.items()
            }

    def _remove_nans(self, key: str, value: torch.Tensor):
        if self.nan_masks is None:
            raise ValueError(
                "record_batch must be called at least once before removing NaNs"
            )
        nan_mask = self.nan_masks[key]
        if nan_mask is not None:
            mask = ~nan_mask.to(value.device).expand(*value.shape)
            value = torch.masked_select(value, mask)
        else:
            value = value.flatten()
        return value

    @torch.no_grad()
    def record_batch(self, data: TensorMapping):
        self.set_nan_masks(data)
        if self.histograms is None:
            self.histograms = {
                k: DynamicHistogram(n_series=1, n_bins=self.n_bins) for k in data.keys()
            }
        for k in data:
            # no matter what data shape is given, combine it all into one histogram
            value = self._remove_nans(k, data[k])
            self.histograms[k].add([value])

    def get_histograms(self) -> dict[str, _Histogram]:
        if self.histograms is None:
            raise ValueError("No data has been added to the histogram")
        return_dict: dict[str, _Histogram] = {}
        for k, histogram in self.histograms.items():
            counts, bin_edges = trim_zero_bins(
                histogram.counts[0, :], histogram.get_bin_edges()
            )
            return_dict[k] = _Histogram(counts, bin_edges)
        return return_dict

    def get_dataset(self) -> xr.Dataset:
        if self.histograms is None:
            raise ValueError("No data has been added to the histogram")
        data = {}
        for var_name, histogram in self.histograms.items():
            data[var_name] = xr.DataArray(
                histogram.counts[0, :],
                dims=("bin",),
            )
            data[f"{var_name}_bin_edges"] = xr.DataArray(
                histogram.get_bin_edges(),
                dims=("bin_edges",),
            )
        return xr.Dataset(data)

    def _plot_histogram(self, histogram: _Histogram) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots()
        normalized_counts = _normalize_histogram(histogram.counts, histogram.bin_edges)
        ax.step(
            histogram.bin_edges[:-1],
            normalized_counts,
            "-",
            where="post",
        )
        ax.set(yscale="log")
        plt.tight_layout()
        return fig

    def get_wandb(
        self, prefix: str = ""
    ) -> dict[str, float | matplotlib.figure.Figure]:
        return_dict = {}
        histograms = self.get_histograms()
        prefix = _add_trailing_slash(prefix)

        for field_name, histogram in histograms.items():
            fig = self._plot_histogram(histogram)
            return_dict[field_name] = fig
            plt.close(fig)
            for p in self.percentiles:
                return_dict[f"{prefix}{p}th-percentile/{field_name}"] = quantile(
                    histogram.bin_edges, histogram.counts, p / 100.0
                )
        return return_dict


class ComparedDynamicHistograms:
    """Wrapper of DynamicHistogram for multiple histograms, two histograms per
    variable plotted on the same axis.
    """

    def __init__(
        self,
        n_bins: int,
        percentiles: list[float] | None = None,
        compute_percentile_frac: bool = False,
    ) -> None:
        self.n_bins = n_bins
        percentiles = [99.9999] if percentiles is None else percentiles
        self.percentiles = [p for p in percentiles]
        self.target_aggregator: DynamicHistogramAggregator | None = None
        self.prediction_aggregator: DynamicHistogramAggregator | None = None
        self._nan_masks: Mapping[str, torch.Tensor] | None = None
        self._variables: set[str] = set()
        self._compute_percentile_frac = compute_percentile_frac

    def _check_overlapping_keys(self, target: TensorMapping, prediction: TensorMapping):
        if not self._variables:
            self._variables = set(target.keys()).intersection(prediction.keys())
        if not self._variables:
            raise ValueError(
                "No overlapping keys between target and prediction variables. "
                f"target: {target.keys()}, prediction: {prediction.keys()}"
            )

        current_variables = set(target.keys()).intersection(prediction.keys())
        if current_variables != self._variables:
            raise ValueError(
                "Available comparison variables provided to record_batch differ "
                f"from initial call to record_batch.  initial: {self._variables}, "
                f"current: {current_variables}"
            )

    def _initialize_histogram_aggs(self, target_data: TensorMapping):
        if self._variables is None:
            raise RuntimeError(
                "_check_overlapping_keys must be called to get variable set "
                "before initializing histograms."
            )
        if self.target_aggregator is None:
            self.target_aggregator = DynamicHistogramAggregator(
                n_bins=self.n_bins, percentiles=self.percentiles
            )
            self.target_aggregator.set_nan_masks(target_data)
        if self.prediction_aggregator is None:
            self.prediction_aggregator = DynamicHistogramAggregator(
                n_bins=self.n_bins,
                percentiles=self.percentiles,
                # nan_masks are defined using the target data
                nan_masks=self.target_aggregator.nan_masks,
            )

    @torch.no_grad()
    def record_batch(self, target: TensorMapping, prediction: TensorMapping):
        self._check_overlapping_keys(target, prediction)
        target = {k: v for k, v in target.items() if k in self._variables}
        prediction = {k: v for k, v in prediction.items() if k in self._variables}
        self._initialize_histogram_aggs(target_data=target)
        assert self.target_aggregator is not None
        self.target_aggregator.record_batch(target)
        assert self.prediction_aggregator is not None
        self.prediction_aggregator.record_batch(prediction)

    def _get_histograms(
        self,
    ) -> dict[str, dict[Literal["target", "prediction"], _Histogram]]:
        if self.target_aggregator is None or self.prediction_aggregator is None:
            raise ValueError("No data has been added to the histogram")
        return_dict: dict[str, dict[Literal["target", "prediction"], _Histogram]] = (
            collections.defaultdict(dict)
        )
        target_histograms = self.target_aggregator.get_histograms()
        prediction_histograms = self.prediction_aggregator.get_histograms()
        for k in self._variables:
            return_dict[k]["target"] = target_histograms[k]
            return_dict[k]["prediction"] = prediction_histograms[k]
        return return_dict

    def _plot_histogram(
        self, target_histogram: _Histogram, prediction_histogram: _Histogram
    ) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots()
        for histogram, label, line_style, color in zip(
            (target_histogram, prediction_histogram),
            ("target", "prediction"),
            ("-", "--"),
            ("C0", "C1"),
        ):
            normalized_counts = _normalize_histogram(
                histogram.counts, histogram.bin_edges
            )
            ax.step(
                histogram.bin_edges[:-1],
                normalized_counts,
                line_style,
                where="post",
                label=label,
                color=color,
            )
        ax.set(yscale="log")
        ax.legend()
        plt.tight_layout()
        return fig

    def get_wandb(
        self, percentile_variables: set[str] | None = None
    ) -> dict[str, float]:
        return_dict: dict[str, matplotlib.figure.Figure | float] = {}

        for field_name, histograms in self._get_histograms().items():
            target = histograms["target"]
            prediction = histograms["prediction"]
            fig = self._plot_histogram(target, prediction)
            return_dict[field_name] = fig
            plt.close(fig)
            emit_percentiles = (
                percentile_variables is None or field_name in percentile_variables
            )
            if emit_percentiles:
                for p in self.percentiles:
                    return_dict[f"target/{p}th-percentile/{field_name}"] = quantile(
                        target.bin_edges, target.counts, p / 100.0
                    )
                    return_dict[f"prediction/{p}th-percentile/{field_name}"] = quantile(
                        prediction.bin_edges, prediction.counts, p / 100.0
                    )
                    if self._compute_percentile_frac:
                        return_dict[
                            f"prediction_frac_of_target/{p}th-percentile/{field_name}"
                        ] = (
                            return_dict[f"prediction/{p}th-percentile/{field_name}"]
                            / return_dict[f"target/{p}th-percentile/{field_name}"]
                        )

        return return_dict

    def get_dataset(self) -> xr.Dataset:
        if self.target_aggregator is None or self.prediction_aggregator is None:
            raise ValueError("No data has been added to the histogram")
        target_dataset = self.target_aggregator.get_dataset()
        prediction_dataset = self.prediction_aggregator.get_dataset()
        for missing_target_name in set(prediction_dataset.data_vars) - set(
            target_dataset.data_vars
        ):
            if not missing_target_name.endswith("_bin_edges"):
                target_dataset[missing_target_name] = xr.DataArray(
                    np.zeros_like(prediction_dataset[missing_target_name]),
                    dims=("bin",),
                )
                target_dataset[f"{missing_target_name}_bin_edges"] = prediction_dataset[
                    f"{missing_target_name}_bin_edges"
                ]
        for missing_prediction_name in set(target_dataset.data_vars) - set(
            prediction_dataset.data_vars
        ):
            if not missing_prediction_name.endswith("_bin_edges"):
                prediction_dataset[missing_prediction_name] = xr.DataArray(
                    np.zeros_like(target_dataset[missing_prediction_name]),
                    dims=("bin",),
                )
                prediction_dataset[f"{missing_prediction_name}_bin_edges"] = (
                    target_dataset[f"{missing_prediction_name}_bin_edges"]
                )
        ds = xr.concat([target_dataset, prediction_dataset], dim="source")
        ds["source"] = ["target", "prediction"]
        return ds


class ComparedDynamicTailsHistograms(ComparedDynamicHistograms):
    """ComparedDynamicHistograms with support for upper-tailed, lower-tailed,
    and two-tailed distribution metrics. By default, variables not in either list
    two_tailed_variables or left_tailed_variables are treated as upper-tailed.

    Percentiles values should be specified as the percentile of the upper tail.
    This class will automatically compute the complementary percentile for the
    lower tail.
    """

    def __init__(
        self,
        n_bins: int,
        percentiles: list[float] | None = None,
        compute_percentile_frac: bool = True,
        two_tailed_variables: list[str] | None = None,
        left_tailed_variables: list[str] | None = None,
        right_tailed_variables: list[str] | None = None,
        default_tail: Literal["upper", "lower", "both"] = "upper",
    ) -> None:
        super().__init__(
            n_bins=n_bins,
            percentiles=percentiles,
            compute_percentile_frac=compute_percentile_frac,
        )
        self._two_tailed_variables = two_tailed_variables or []
        self._left_tailed_variables = left_tailed_variables or []
        self._right_tailed_variables = right_tailed_variables or []
        self._default_tail = default_tail

    def _variable_distribution_tail(
        self, field_name: str
    ) -> Literal["upper", "lower", "both"]:
        if field_name in self._two_tailed_variables:
            return "both"
        if field_name in self._left_tailed_variables:
            return "lower"
        if field_name in self._right_tailed_variables:
            return "upper"
        return self._default_tail

    def _get_percentile_metrics_for_field(
        self, histogram: _Histogram, field_name: str
    ) -> dict[str, float]:
        tail = self._variable_distribution_tail(field_name)
        percentile_metrics: dict[str, float] = {}
        for p in self.percentiles:
            if tail in ["upper", "both"]:
                p_key = _format_percentile_for_metric_key(p, p)
                percentile_metrics[f"{p_key}th-percentile/{field_name}"] = quantile(
                    histogram.bin_edges, histogram.counts, p / 100.0
                )
            if tail in ["lower", "both"]:
                p_key = _format_percentile_for_metric_key(100.0 - p, p)
                percentile_metrics[f"{p_key}th-percentile/{field_name}"] = quantile(
                    histogram.bin_edges, histogram.counts, (100.0 - p) / 100.0
                )
        return percentile_metrics

    def get_wandb(
        self, percentile_variables: set[str] | None = None
    ) -> dict[str, float]:
        return_metrics: dict[str, matplotlib.figure.Figure | float] = {}
        target_metrics: dict[str, float] = {}

        for field_name, histograms in self._get_histograms().items():
            target = histograms["target"]
            prediction = histograms["prediction"]
            fig = self._plot_histogram(target, prediction)
            return_metrics[field_name] = fig
            plt.close(fig)
            emit_percentiles = (
                percentile_variables is None or field_name in percentile_variables
            )
            if emit_percentiles:
                return_metrics.update(
                    self._get_percentile_metrics_for_field(prediction, field_name)
                )

                if self._compute_percentile_frac:
                    target_metrics.update(
                        self._get_percentile_metrics_for_field(
                            target,
                            field_name,
                        )
                    )
        for key, value in target_metrics.items():
            return_metrics[f"prediction_frac_of_target/{key}"] = (
                return_metrics[key] / value
            )
        return return_metrics


def _normalize_histogram(counts, bin_edges):
    """
    Normalize histogram counts so that the integral is 1.
    """
    bin_widths = np.diff(bin_edges)
    return counts / np.sum(counts * bin_widths)
