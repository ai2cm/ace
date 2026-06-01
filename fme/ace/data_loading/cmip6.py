import dataclasses
import logging
import os
from collections.abc import Sequence
from typing import Literal

import pandas as pd

from fme.core.dataset.concat import ConcatDatasetConfig
from fme.core.dataset.config import DatasetConfigABC
from fme.core.dataset.dataset import DatasetABC
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.time import TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Cmip6TimeMask:
    """A contiguous time window to drop from datasets matching the
    ``source_ids`` × ``experiments`` filter.

    Designed for the temporal-interpolation holdout pattern: leave a
    gap in the middle of a model's historical record so training data
    spans ``[earliest, keep_before] ∪ [keep_after, latest]`` and the
    gap (``(keep_before, keep_after)``) can be used as eval. The
    masked dataset is split into a pre-mask and a post-mask
    ``XarrayDataConfig`` entry; the existing ``ConcatDatasetConfig``
    machinery concatenates the two, so the masked region is invisible
    to the trainer.

    Calendar-independent — boundaries are passed straight to xarray's
    ``TimeSlice``, which handles whatever calendar the underlying
    dataset uses (no day arithmetic on this side, so a 360-day
    calendar produces the same masking as a noleap one).

    Parameters:
        source_ids: Source models the mask applies to.
        experiments: Experiments the mask applies to (only datasets
            matching both filters are masked).
        keep_before: Inclusive last date kept in the pre-mask slice
            (e.g. ``"1969-12-31"`` to start the gap on 1970-01-01).
        keep_after: Inclusive first date kept in the post-mask slice
            (e.g. ``"1990-01-01"`` to end the gap on 1989-12-31).
    """

    source_ids: list[str]
    experiments: list[str]
    keep_before: str
    keep_after: str

    def matches(self, source_id: str, experiment: str) -> bool:
        return source_id in self.source_ids and experiment in self.experiments


@dataclasses.dataclass
class Cmip6DataConfig(DatasetConfigABC):
    """Configuration for loading CMIP6 processed datasets.

    Reads index.csv from data_dir, filters to matching datasets, and
    delegates to ConcatDatasetConfig for the actual data loading.
    Each (source_id, experiment, variant_label) combination produces
    one XarrayDataConfig entry (two when a ``time_masks`` entry
    matches — pre-mask + post-mask).

    Parameters:
        data_dir: Path to the directory containing index.csv and data files.
        source_ids: Source model IDs to include. If None, all available.
        exclude_source_ids: Source model IDs to drop wholesale.
        experiments: Experiment IDs to include.
        realizations: Realization numbers (r values) to include. If None, all.
        exclude_variants: Specific
            ``[source_id, experiment, variant_label]`` triples to drop
            after the include/exclude filters above. Use for
            per-(source, experiment, variant) holdouts (e.g. drop one
            r2 of MRI-ESM2-0/historical) where the broader filters can't
            target a single variant.
        time_masks: Time-window cutouts applied per matched
            (source_id, experiment) pair. Each matched dataset gets
            split into a pre-mask and a post-mask XarrayDataConfig
            entry. Only one mask per (source_id, experiment) is
            supported; overlapping configurations raise at
            ``__post_init__``.
        engine: Xarray engine for reading data files. "zarr" reads data.zarr
            stores, "netcdf4" reads data.nc files. Use "netcdf4" with
            zarr_to_netcdf.py-converted datasets for fork-safe data workers.
    """

    data_dir: str
    source_ids: list[str] | None = None
    exclude_source_ids: list[str] = dataclasses.field(default_factory=list)
    experiments: list[str] = dataclasses.field(
        default_factory=lambda: ["historical", "ssp585"]
    )
    realizations: list[int] | None = None
    exclude_variants: list[list[str]] = dataclasses.field(default_factory=list)
    time_masks: list[Cmip6TimeMask] = dataclasses.field(default_factory=list)
    engine: Literal["zarr", "netcdf4"] = "zarr"

    def __post_init__(self):
        self._concat_config_cache: ConcatDatasetConfig | None = None
        if self.engine not in ("zarr", "netcdf4"):
            raise ValueError(f"engine must be 'zarr' or 'netcdf4', got {self.engine!r}")
        for triple in self.exclude_variants:
            if len(triple) != 3:
                raise ValueError(
                    "Each exclude_variants entry must be a "
                    "[source_id, experiment, variant_label] triple; "
                    f"got {triple!r}"
                )
        # Disallow overlapping time_masks on the same (source, experiment) —
        # the pre/post split this class produces assumes a single mask
        # window per dataset.
        seen: set[tuple[str, str]] = set()
        for mask in self.time_masks:
            for src in mask.source_ids:
                for exp in mask.experiments:
                    key = (src, exp)
                    if key in seen:
                        raise ValueError(
                            f"Multiple time_masks match ({src!r}, {exp!r}); "
                            "only one mask per (source_id, experiment) is "
                            "supported."
                        )
                    seen.add(key)

    @property
    def _file_pattern(self) -> str:
        return "data.zarr" if self.engine == "zarr" else "data.*.nc"

    @property
    def zarr_engine_used(self) -> bool:
        return self.engine == "zarr"

    def _get_concat_config(self) -> ConcatDatasetConfig:
        if self._concat_config_cache is None:
            self._concat_config_cache = self._build_concat_config()
        return self._concat_config_cache

    def _load_and_filter_index(self) -> pd.DataFrame:
        index_path = os.path.join(self.data_dir, "index.csv")
        idx = pd.read_csv(index_path)
        mask = idx["status"] == "ok"
        if self.source_ids is not None:
            mask &= idx["source_id"].isin(self.source_ids)
        if self.exclude_source_ids:
            mask &= ~idx["source_id"].isin(self.exclude_source_ids)
        mask &= idx["experiment"].isin(self.experiments)
        if self.realizations is not None:
            mask &= idx["variant_r"].isin(self.realizations)
        filtered = idx[mask]
        # exclude_variants is applied after the include filters: it's
        # subtractive ("drop this specific row"), so rows ruled out
        # earlier can't be silently re-included by listing them here.
        if self.exclude_variants:
            excluded = {tuple(t) for t in self.exclude_variants}
            keep_mask = ~filtered.apply(
                lambda r: (r["source_id"], r["experiment"], r["variant_label"])
                in excluded,
                axis=1,
            )
            n_dropped = (~keep_mask).sum()
            filtered = filtered[keep_mask]
            unseen = excluded - {
                (r["source_id"], r["experiment"], r["variant_label"])
                for _, r in idx.iterrows()
            }
            if unseen:
                raise ValueError(
                    "exclude_variants triples were not present in index.csv: "
                    f"{sorted(unseen)}"
                )
            logger.info(
                "Cmip6DataConfig: dropped %d datasets via exclude_variants",
                n_dropped,
            )
        if len(filtered) == 0:
            raise ValueError(
                f"No datasets with status='ok' matched the filter criteria "
                f"in {index_path}. "
                f"source_ids={self.source_ids}, "
                f"experiments={self.experiments}, "
                f"realizations={self.realizations}, "
                f"exclude_variants={self.exclude_variants}"
            )
        return filtered

    def _find_time_mask(self, source_id: str, experiment: str) -> Cmip6TimeMask | None:
        for mask in self.time_masks:
            if mask.matches(source_id, experiment):
                return mask
        return None

    def _build_concat_config(self) -> ConcatDatasetConfig:
        idx = self._load_and_filter_index()
        configs: list[XarrayDataConfig] = []
        n_masked = 0
        for _, row in idx.iterrows():
            data_path = os.path.join(
                self.data_dir,
                row["source_id"],
                row["experiment"],
                row["variant_label"],
            )
            mask = self._find_time_mask(row["source_id"], row["experiment"])
            if mask is None:
                configs.append(
                    XarrayDataConfig(
                        data_path=data_path,
                        file_pattern=self._file_pattern,
                        engine=self.engine,
                        labels=[row["label"]],
                    )
                )
                continue
            # Split this dataset into pre-mask and post-mask entries.
            # Both carry the same label so per-source normalization and
            # any other label-keyed downstream behaviour treats them
            # as the same dataset.
            configs.append(
                XarrayDataConfig(
                    data_path=data_path,
                    file_pattern=self._file_pattern,
                    engine=self.engine,
                    labels=[row["label"]],
                    subset=TimeSlice(stop_time=mask.keep_before),
                )
            )
            configs.append(
                XarrayDataConfig(
                    data_path=data_path,
                    file_pattern=self._file_pattern,
                    engine=self.engine,
                    labels=[row["label"]],
                    subset=TimeSlice(start_time=mask.keep_after),
                )
            )
            n_masked += 1
        logger.info(
            "Cmip6DataConfig: %d datasets from %d source models "
            "(%d masked into pre/post slices)",
            len(configs),
            idx["source_id"].nunique(),
            n_masked,
        )
        return ConcatDatasetConfig(concat=configs, strict=False)

    @property
    def available_labels(self) -> set[str] | None:
        return self._get_concat_config().available_labels

    def build(
        self,
        names: Sequence[str],
        n_timesteps: IntSchedule,
        allow_missing_variables: bool = False,
    ) -> tuple[DatasetABC, DatasetProperties]:
        return self._get_concat_config().build(
            names, n_timesteps, allow_missing_variables=allow_missing_variables
        )
