import collections
from collections import namedtuple
from typing import Dict, List, Literal, Mapping, Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

from fme.core.metrics import quantile
from fme.core.typing_ import TensorMapping

EPSILON = 1.0e-6


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
        self.bin_edges: Optional[np.ndarray] = None
        self.counts = np.zeros((n_times, n_bins), dtype=np.int64)
        self._epsilon: float = EPSILON

    def add(self, value: np.ndarray, i_time_start: int = 0):
        """
        Add new values to the histogram.

        Args:
            value: array of values of shape (n_times, n_values) to add to the histogram
            i_time_start: index of the first time to add values to
        """
        vmin = np.min(value)
        vmax = np.max(value)
        if vmin == vmax:
            # if all values are the same, add a small amount to vmin and vmax
            vmin -= self._epsilon
            vmax += self._epsilon
        if self.bin_edges is None:
            self.bin_edges = np.linspace(vmin, vmax, self._n_bins + 1)
        else:
            while vmin < self.bin_edges[0]:
                self._double_size_left()
            while vmax > self.bin_edges[-1]:
                self._double_size_right()
        i_time_end = i_time_start + value.shape[0]
        self.counts[i_time_start:i_time_end, :] += np.apply_along_axis(
            lambda arr: np.histogram(arr, bins=self.bin_edges)[0],
            axis=1,
            arr=value,
        )

    def _double_size_left(self):
        """
        Double the sizes of bins, extending the histogram
        to the left (further negative).
        """
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
    variable plotted on the same axis."""

    def __init__(self, n_bins: int, percentiles: Optional[List[float]] = None) -> None:
        self.n_bins = n_bins
        percentiles = [99.9999] if percentiles is None else percentiles
        self.percentiles = [p for p in percentiles]
        self.target_histograms: Optional[Mapping[str, DynamicHistogram]] = None
        self.prediction_histograms: Optional[Mapping[str, DynamicHistogram]] = None
        self._time_dim = -2

    @torch.no_grad()
    def record_batch(self, target: TensorMapping, prediction: TensorMapping):
        target = {k: v.detach().cpu() for k, v in target.items()}
        prediction = {k: v.detach().cpu() for k, v in prediction.items()}

        if self.target_histograms is None or self.prediction_histograms is None:
            self.target_histograms = {}
            for k in target:
                self.target_histograms[k] = DynamicHistogram(
                    n_times=1, n_bins=self.n_bins
                )
            self.prediction_histograms = {}
            for k in prediction:
                self.prediction_histograms[k] = DynamicHistogram(
                    n_times=1, n_bins=self.n_bins
                )

        for k in target:
            # no matter what data shape is given, combine it all into one histogram
            self.target_histograms[k].add(target[k].flatten().unsqueeze(0).numpy())
        for k in prediction:
            self.prediction_histograms[k].add(
                prediction[k].flatten().unsqueeze(0).numpy()
            )

    def _get_histograms(
        self,
    ) -> Dict[str, Dict[Literal["target", "prediction"], _Histogram]]:
        if self.target_histograms is None or self.prediction_histograms is None:
            raise ValueError("No data has been added to the histogram")
        return_dict: Dict[
            str, Dict[Literal["target", "prediction"], _Histogram]
        ] = collections.defaultdict(dict)
        for k in self.target_histograms:
            return_dict[k]["target"] = _Histogram(
                self.target_histograms[k].counts.squeeze(self._time_dim),
                self.target_histograms[k].bin_edges,
            )
        for k in self.prediction_histograms:
            return_dict[k]["prediction"] = _Histogram(
                self.prediction_histograms[k].counts.squeeze(self._time_dim),
                self.prediction_histograms[k].bin_edges,
            )
        return return_dict

    def get(self):
        """Returns a dict containing histograms for target and
        prediction."""
        return_dict = {}
        for field_name, metrics_dict in self._get_histograms().items():
            if "target" in metrics_dict:
                target = metrics_dict["target"]
                return_dict[f"target/{field_name}"] = target
                for p in self.percentiles:
                    return_dict[f"target/{p}th-percentile/{field_name}"] = quantile(
                        target.bin_edges, target.counts, p / 100.0
                    )
            if "prediction" in metrics_dict:
                prediction = metrics_dict["prediction"]
                return_dict[f"prediction/{field_name}"] = prediction
                for p in self.percentiles:
                    return_dict[f"prediction/{p}th-percentile/{field_name}"] = quantile(
                        prediction.bin_edges, prediction.counts, p / 100.0
                    )
        return return_dict

    def _plot_histogram(
        self, target_histogram: Optional[_Histogram], prediction_histogram
    ) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots()
        for histogram, label, line_style, color in zip(
            (target_histogram, prediction_histogram),
            ("target", "prediction"),
            ("-", "--"),
            ("C0", "C1"),
        ):
            if histogram is not None:
                ax.step(
                    histogram.bin_edges[:-1],
                    histogram.counts,
                    line_style,
                    where="post",
                    label=label,
                    color=color,
                )
        ax.set(yscale="log")
        ax.legend()
        plt.tight_layout()
        return fig

    def get_wandb(self) -> Dict[str, float]:
        return_dict: Dict[str, float] = {}

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
        return return_dict
