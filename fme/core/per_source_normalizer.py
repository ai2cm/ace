import dataclasses
import logging
import os

import torch

from fme.core.labels import BatchLabels
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.typing_ import TensorDict, TensorMapping

logger = logging.getLogger(__name__)


class PerSourceNormalizer:
    """Normalizer that selects per-label normalization constants.

    Each batch element is normalized using the constants associated with
    its label. Unknown labels fall back to the default normalizer.
    """

    def __init__(
        self,
        normalizers: dict[str, StandardNormalizer],
        default: StandardNormalizer,
    ):
        self._normalizers = normalizers
        self._default = default

    @property
    def default(self) -> StandardNormalizer:
        return self._default

    def normalize(
        self, tensors: TensorMapping, labels: BatchLabels | None = None
    ) -> TensorDict:
        if labels is None or not self._normalizers:
            return self._default.normalize(tensors)
        return self._apply(tensors, labels, normalize=True)

    def denormalize(
        self, tensors: TensorMapping, labels: BatchLabels | None = None
    ) -> TensorDict:
        if labels is None or not self._normalizers:
            return self._default.denormalize(tensors)
        return self._apply(tensors, labels, normalize=False)

    def _apply(
        self, tensors: TensorMapping, labels: BatchLabels, normalize: bool
    ) -> TensorDict:
        label_indices = labels.tensor.argmax(dim=1)
        unique_indices = label_indices.unique()

        if len(unique_indices) == 1:
            label_name = labels.names[unique_indices[0].item()]
            normalizer = self._normalizers.get(label_name, self._default)
            if normalize:
                return normalizer.normalize(tensors)
            return normalizer.denormalize(tensors)

        names = self._default._names
        filtered = {k: v for k, v in tensors.items() if k in names}
        result = {k: torch.empty_like(v) for k, v in filtered.items()}

        for idx in unique_indices:
            mask = label_indices == idx
            label_name = labels.names[idx.item()]
            normalizer = self._normalizers.get(label_name, self._default)
            subset = {k: v[mask] for k, v in filtered.items()}
            if normalize:
                transformed = normalizer.normalize(subset)
            else:
                transformed = normalizer.denormalize(subset)
            for k, v in transformed.items():
                result[k][mask] = v

        return result


@dataclasses.dataclass
class PerSourceNormalizationConfig:
    """Configuration for per-source-model normalization.

    Supports two modes:
    - File-based: set ``data_dir`` to the CMIP6 data directory. Per-source
      stats are loaded from ``{data_dir}/{subdirectory}/{label}/centering.nc``
      and ``scaling.nc``.
    - Explicit: set ``sources`` to a dict mapping label names to
      ``NormalizationConfig`` instances (used after ``load()`` and in tests).

    Parameters:
        data_dir: Root data directory (same as ``Cmip6DataConfig.data_dir``).
        subdirectory: Subdirectory within data_dir containing per-source stats.
        means_filename: Filename for centering statistics within each label dir.
        stds_filename: Filename for scaling statistics within each label dir.
        sources: Explicit per-label normalization configs (populated by load).
    """

    data_dir: str | None = None
    subdirectory: str = "per_source_normalization"
    means_filename: str = "centering.nc"
    stds_filename: str = "scaling.nc"
    sources: dict[str, NormalizationConfig] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.data_dir is None and not self.sources:
            raise ValueError(
                "PerSourceNormalizationConfig requires either data_dir or "
                "explicit sources."
            )

    def load(self):
        if self.data_dir is not None:
            norm_dir = os.path.join(self.data_dir, self.subdirectory)
            if not os.path.isdir(norm_dir):
                raise FileNotFoundError(
                    f"Per-source normalization directory not found: {norm_dir}"
                )
            for entry in sorted(os.scandir(norm_dir), key=lambda e: e.name):
                if not entry.is_dir():
                    continue
                means_path = os.path.join(entry.path, self.means_filename)
                stds_path = os.path.join(entry.path, self.stds_filename)
                if os.path.exists(means_path) and os.path.exists(stds_path):
                    config = NormalizationConfig(
                        global_means_path=means_path,
                        global_stds_path=stds_path,
                    )
                    config.load()
                    self.sources[entry.name] = config
            logger.info(
                "Loaded per-source normalization for %d labels from %s",
                len(self.sources),
                norm_dir,
            )
            self.data_dir = None

    def build(
        self, names: list[str], default: StandardNormalizer
    ) -> PerSourceNormalizer:
        normalizers = {}
        for label, config in self.sources.items():
            normalizers[label] = config.build(names)
        return PerSourceNormalizer(normalizers=normalizers, default=default)
