from typing import Optional

import numpy as np


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

    def add(self, value: np.ndarray, i_time_start: int = 0):
        """
        Add new values to the histogram.

        Args:
            value: array of values of shape (n_times, n_values) to add to the histogram
            i_time_start: index of the first time to add values to
        """
        vmin = np.min(value)
        vmax = np.max(value)
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
