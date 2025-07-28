import collections
import logging
from collections import namedtuple
from collections.abc import Mapping
from typing import Literal

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from fme.core.metrics import quantile
from fme.core.typing_ import TensorMapping

EPSILON = 1.0e-6


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
    Compute histogram of a tensor.

    Args:
        arr: 1D tensor of values
        vmin: minimum value of the histogram
        bin_size: size of each bin
        n_bins: number of bins

    Returns:
        counts, bin_edges: arrays of counts and bin edges
    """
    # use np.bincount for speed
    counts = torch.bincount(((arr - vmin) / bin_size).int(), minlength=n_bins)
    return_counts = counts[:n_bins]
    return_counts[-1] += torch.sum(counts[n_bins:])
    return return_counts


class DynamicHistogram:
    """
    A histogram that dynamically bins values into a fixed number of bins
    of constant width. A separate histogram is defined for each time,
    and the same bins are used for all times.

    When a new value is added that goes out of range of the current bins,
    bins are doubled in size until that value is within the range of the bins.
    """

    def __init__(self, n_times: int, n_bins: int = 300):
        """
        Args:
            n_times: Length of time dimension of each sample.
            n_bins: Number of bins to use for the histogram. The "effective"
                number of bins may be as small as 1/4th this number of bins,
                as there may be bins greater than the max or less than the
                min value with no samples in them, due to the dynamic resizing
                of bins.
        """
        self._n_times = n_times
        self._n_bins = n_bins
        self.bin_edges: np.ndarray | None = None
        self.counts = np.zeros((n_times, n_bins), dtype=np.int64)
        self._epsilon: float = EPSILON

    def add(self, value: torch.Tensor, i_time_start: int = 0):
        """
        Add new values to the histogram.

        Args:
            value: tensor of values of shape (n_times, n_values) to add to the histogram
            i_time_start: index of the first time to add values to
        """
        # add epsilon to ensure all values stay within (and not just equal to)
        # the bin edges, and to avoid the case where vmin == vmax
        vmin = float((torch.min(value) - self._epsilon).cpu().numpy())
        vmax = float((torch.max(value) + self._epsilon).cpu().numpy())

        if self.bin_edges is None:
            self.bin_edges = np.linspace(vmin, vmax, self._n_bins + 1)
        else:
            while vmin < self.bin_edges[0]:
                self._double_size_left()
            while vmax > self.bin_edges[-1]:
                self._double_size_right()

        i_time_end = i_time_start + value.shape[0]
        for i_time in range(i_time_start, i_time_end):
            try:
                self.counts[i_time] += (
                    histogram(
                        value[i_time - i_time_start, :],
                        self.bin_edges[0],
                        self.bin_edges[1] - self.bin_edges[0],
                        self._n_bins,
                    )
                    .cpu()
                    .numpy()
                )
            except RuntimeError as err:
                # ignore samples with NaNs
                logging.error(f"caught exception while computing histogram: {err}")

    def _double_size_left(self):
        """
        Double the sizes of bins, extending the histogram
        to the left (further negative).
        """
        if self.bin_edges is None:
            raise RuntimeError("Cannot double size of bins without bin edges")
        current_range = self.bin_edges[-1] - self.bin_edges[0]
        new_range = 2 * current_range

        new_bin_edges = np.linspace(
            self.bin_edges[-1] - new_range,
            self.bin_edges[-1],
            self._n_bins + 1,
        )
        new_counts = np.zeros((self._n_times, self._n_bins), dtype=np.int64)
        combined_counts = self.counts[:, ::2] + self.counts[:, 1::2]
        new_counts[:, self._n_bins // 2 :] = combined_counts
        self.bin_edges = new_bin_edges
        self.counts = new_counts

    def _double_size_right(self):
        """
        Double the sizes of bins, extending the histogram
        to the right (further positive).
        """
        if self.bin_edges is None:
            raise RuntimeError("Cannot double size of bins without bin edges")
        current_range = self.bin_edges[-1] - self.bin_edges[0]
        new_range = 2 * current_range

        new_bin_edges = np.linspace(
            self.bin_edges[0],
            self.bin_edges[0] + new_range,
            self._n_bins + 1,
        )
        new_counts = np.zeros((self._n_times, self._n_bins), dtype=np.int64)
        combined_counts = self.counts[:, ::2] + self.counts[:, 1::2]
        new_counts[:, : self._n_bins // 2] = combined_counts
        self.bin_edges = new_bin_edges
        self.counts = new_counts


_Histogram = namedtuple("_Histogram", ["counts", "bin_edges"])


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
        self.target_histograms: Mapping[str, DynamicHistogram] | None = None
        self.prediction_histograms: Mapping[str, DynamicHistogram] | None = None
        self._nan_masks: Mapping[str, torch.Tensor] | None = None
        self._time_dim = -2
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

    @torch.no_grad()
    def record_batch(self, target: TensorMapping, prediction: TensorMapping):
        self._check_overlapping_keys(target, prediction)

        target = {k: v for k, v in target.items() if k in self._variables}
        prediction = {k: v for k, v in prediction.items() if k in self._variables}
        if self._nan_masks is None:
            # assume the final two dimensions are the spatial dimensions and the
            # spatial pattern of NaNs is the same for all samples & time, but
            # can vary by variable
            ndim = next(iter(target.values())).ndim
            index = (slice(0, 1),) * (ndim - 2) + (slice(None), slice(None))
            self._nan_masks = {
                k: v[index].isnan() if torch.any(v[index].isnan()) else None
                for k, v in target.items()
                if k in self._variables
            }

        if self.target_histograms is None or self.prediction_histograms is None:
            self.target_histograms = {}
            for k in target:
                self.target_histograms[k] = DynamicHistogram(
                    n_times=1,
                    n_bins=self.n_bins,
                )
            self.prediction_histograms = {}
            for k in prediction:
                self.prediction_histograms[k] = DynamicHistogram(
                    n_times=1,
                    n_bins=self.n_bins,
                )
        for k in target:
            # no matter what data shape is given, combine it all into one histogram
            value = self._remove_nans(k, target[k])
            self.target_histograms[k].add(value)
        for k in prediction:
            value = self._remove_nans(k, prediction[k])
            self.prediction_histograms[k].add(value)

    def _remove_nans(self, key: str, value: torch.Tensor):
        if self._nan_masks is None:
            raise ValueError(
                "record_batch must be called at least once before removing NaNs"
            )
        nan_mask = self._nan_masks[key]
        if nan_mask is not None:
            mask = ~nan_mask.to(value.device).expand(*value.shape)
            value = torch.masked_select(value, mask)
        else:
            value = value.flatten()
        return value.unsqueeze(0)

    def _get_histograms(
        self,
    ) -> dict[str, dict[Literal["target", "prediction"], _Histogram]]:
        if self.target_histograms is None or self.prediction_histograms is None:
            raise ValueError("No data has been added to the histogram")
        return_dict: dict[str, dict[Literal["target", "prediction"], _Histogram]] = (
            collections.defaultdict(dict)
        )
        for k in self.target_histograms:
            counts, bin_edges = trim_zero_bins(
                self.target_histograms[k].counts.squeeze(self._time_dim),
                self.target_histograms[k].bin_edges,
            )
            return_dict[k]["target"] = _Histogram(counts, bin_edges)
        for k in self.prediction_histograms:
            counts, bin_edges = trim_zero_bins(
                self.prediction_histograms[k].counts.squeeze(self._time_dim),
                self.prediction_histograms[k].bin_edges,
            )
            return_dict[k]["prediction"] = _Histogram(counts, bin_edges)
        return return_dict

    def _plot_histogram(
        self, target_histogram: _Histogram | None, prediction_histogram
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
            if histogram is not None:
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

    def get_wandb(self) -> dict[str, float]:
        return_dict: dict[str, matplotlib.figure.Figure | float] = {}

        for field_name, histograms in self._get_histograms().items():
            target = histograms.get("target")
            prediction = histograms.get("prediction")
            fig = self._plot_histogram(target, prediction)
            return_dict[field_name] = fig
            plt.close(fig)
            if target is not None:
                for p in self.percentiles:
                    return_dict[f"target/{p}th-percentile/{field_name}"] = quantile(
                        target.bin_edges, target.counts, p / 100.0
                    )
            if prediction is not None:
                for p in self.percentiles:
                    return_dict[f"prediction/{p}th-percentile/{field_name}"] = quantile(
                        prediction.bin_edges, prediction.counts, p / 100.0
                    )
                    if self._compute_percentile_frac and target is not None:
                        return_dict[
                            f"prediction_frac_of_target/{p}th-percentile/{field_name}"
                        ] = (
                            return_dict[f"prediction/{p}th-percentile/{field_name}"]
                            / return_dict[f"target/{p}th-percentile/{field_name}"]
                        )

        return return_dict

    def _get_single_dataset(
        self, histograms: Mapping[str, DynamicHistogram]
    ) -> xr.Dataset:
        data = {}
        for var_name, histogram in histograms.items():
            data[var_name] = xr.DataArray(
                histogram.counts[0, :],
                dims=("bin",),
            )
            data[f"{var_name}_bin_edges"] = xr.DataArray(
                histogram.bin_edges,
                dims=("bin_edges",),
            )
        return xr.Dataset(data)

    def get_dataset(self) -> xr.Dataset:
        if self.target_histograms is None or self.prediction_histograms is None:
            raise ValueError("No data has been added to the histogram")
        target_dataset = self._get_single_dataset(self.target_histograms)
        prediction_dataset = self._get_single_dataset(self.prediction_histograms)
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


def _normalize_histogram(counts, bin_edges):
    """
    Normalize histogram counts so that the integral is 1.
    """
    bin_widths = np.diff(bin_edges)
    return counts / np.sum(counts * bin_widths)
