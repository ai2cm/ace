"""Tendency variance ratio: Var_spatial(pred_tendency) / Var_spatial(tar_tendency).

Monitors whether the model under- or over-represents spatial variability in
its temporal tendency fields.  A ratio < 1 indicates artificially diffuse
tendencies (spectral smoothing); a ratio > 1 indicates over-amplified
spatial variability.
"""

from collections import defaultdict

import torch
import xarray as xr

from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping


class TendencyVarianceAccumulator:
    """Accumulates spatial variance of signed temporal tendencies.

    Call :meth:`record` with paired gen/target data whose second axis is
    time.  The accumulator computes consecutive temporal differences,
    measures spatial variance (over the last two dimensions), and tracks
    running weighted sums so that :meth:`get_ratios` returns the ratio
    of the dataset-level mean spatial variances.
    """

    def __init__(self):
        self._gen_var_sum: dict[str, torch.Tensor] = {}
        self._tgt_var_sum: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = defaultdict(int)

    @torch.no_grad()
    def record(self, gen_data: TensorMapping, target_data: TensorMapping) -> None:
        """Record a batch of paired data with time at axis 1.

        Temporal differences are computed along axis 1; spatial variance
        is taken over the last two dimensions (lat, lon).
        """
        for name in gen_data:
            if name not in target_data:
                continue
            gen = gen_data[name]
            tgt = target_data[name]
            if gen.shape[1] < 2 or tgt.shape[1] < 2:
                continue

            gen_diff = gen[:, 1:] - gen[:, :-1]
            tgt_diff = tgt[:, 1:] - tgt[:, :-1]

            gen_var = gen_diff.var(dim=(-2, -1)).mean()
            tgt_var = tgt_diff.var(dim=(-2, -1)).mean()

            n = gen_diff.shape[0] * gen_diff.shape[1]

            if name not in self._gen_var_sum:
                self._gen_var_sum[name] = gen_var * n
                self._tgt_var_sum[name] = tgt_var * n
            else:
                self._gen_var_sum[name] = self._gen_var_sum[name] + gen_var * n
                self._tgt_var_sum[name] = self._tgt_var_sum[name] + tgt_var * n
            self._counts[name] += n

    def get_ratios(self) -> dict[str, float]:
        """Return per-variable Var(gen tendency) / Var(target tendency)."""
        dist = Distributed.get_instance()
        ratios: dict[str, float] = {}
        for name in sorted(self._gen_var_sum):
            gen_mean = self._gen_var_sum[name] / self._counts[name]
            tgt_mean = self._tgt_var_sum[name] / self._counts[name]
            if dist.world_size > 1:
                gen_mean = dist.reduce_mean(gen_mean)
                tgt_mean = dist.reduce_mean(tgt_mean)
            ratios[name] = float((gen_mean / tgt_mean).cpu())
        return ratios

    def get_logs(self, label: str) -> dict[str, float]:
        """Return logs keyed as ``{label}/tendency_variance_ratio/{var}``."""
        return {
            f"{label}/tendency_variance_ratio/{name}": ratio
            for name, ratio in self.get_ratios().items()
        }

    def get_dataset(self) -> xr.Dataset:
        return xr.Dataset()
