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

    Variables a source doesn't publish (i.e., not in that source's
    ``StandardNormalizer._names``) pass through unchanged for that
    source's batch slice — the data loader populates them as
    NaN-fills under ``allow_missing_variables``, and downstream
    ``fill_nans_on_normalize`` handles the conversion to zeros. The
    optional ``data_mask`` argument lets us assert that any
    pass-through is paired with a False entry in the mask: an
    unmasked sample reaching a slice without per-source stats is a
    config mismatch (per-source norm file genuinely lacks a variable
    the loader thinks is present), and silently passing through
    garbage downstream would be hard to debug — we raise instead.
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
        self,
        tensors: TensorMapping,
        labels: BatchLabels | None = None,
        data_mask: TensorMapping | None = None,
    ) -> TensorDict:
        if labels is None or not self._normalizers:
            return self._default.normalize(tensors)
        return self._apply(tensors, labels, data_mask, normalize=True)

    def denormalize(
        self,
        tensors: TensorMapping,
        labels: BatchLabels | None = None,
        data_mask: TensorMapping | None = None,
    ) -> TensorDict:
        if labels is None or not self._normalizers:
            return self._default.denormalize(tensors)
        return self._apply(tensors, labels, data_mask, normalize=False)

    def _apply(
        self,
        tensors: TensorMapping,
        labels: BatchLabels,
        data_mask: TensorMapping | None,
        normalize: bool,
    ) -> TensorDict:
        label_indices = labels.tensor.argmax(dim=1)
        unique_indices = label_indices.unique()

        # Clone inputs so every key stays in the output and any slice
        # the per-source normalizer doesn't touch (variables the source
        # lacks stats for) passes through unchanged. The prior
        # implementation used ``torch.empty_like`` and selectively
        # filled — silently propagating uninitialized memory whenever
        # a source's normalizer was missing a key.
        result: TensorDict = {k: v.clone() for k, v in tensors.items()}

        for idx in unique_indices:
            source_mask = label_indices == idx
            label_name = labels.names[idx.item()]
            normalizer = self._normalizers.get(label_name, self._default)
            self._apply_for_source(
                normalizer=normalizer,
                source_mask=source_mask,
                label_name=label_name,
                tensors=tensors,
                result=result,
                data_mask=data_mask,
                normalize=normalize,
            )

        return result

    def _apply_for_source(
        self,
        normalizer: StandardNormalizer,
        source_mask: torch.Tensor,
        label_name: str,
        tensors: TensorMapping,
        result: TensorDict,
        data_mask: TensorMapping | None,
        normalize: bool,
    ) -> None:
        """Apply ``normalizer`` to the ``source_mask`` slice of
        ``tensors``, writing into ``result``.

        Variables in ``tensors`` but not in ``normalizer._names``
        (i.e., the source genuinely lacks per-source stats for them)
        skip the lookup entirely; their slice in ``result`` keeps
        the cloned input value. If ``data_mask`` says any sample in
        ``source_mask`` is unmasked for such a variable, raise — a
        config mismatch (loader claims the variable is present but
        the per-source normalization file disagrees) shouldn't
        propagate silently as a NaN/garbage downstream.
        """
        for name in tensors:
            if name in normalizer._names:
                continue
            if data_mask is None:
                continue
            var_mask = data_mask.get(name)
            if var_mask is None:
                continue
            unmasked_in_source = var_mask & source_mask
            if unmasked_in_source.any():
                raise ValueError(
                    f"per-source normalizer for {label_name!r} has no stats for "
                    f"variable {name!r}, but data_mask reports unmasked samples "
                    f"for it in this source's batch slice "
                    f"({int(unmasked_in_source.sum())} samples). Either add "
                    f"stats to the per-source centering/scaling file, or "
                    f"NaN-fill the data so the mask agrees with the stats."
                )

        # Normalize only the variables the source has stats for.
        subset = {
            name: tensors[name][source_mask]
            for name in tensors
            if name in normalizer._names
        }
        if normalize:
            transformed = normalizer.normalize(subset)
        else:
            transformed = normalizer.denormalize(subset)
        for k, v in transformed.items():
            result[k][source_mask] = v


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
