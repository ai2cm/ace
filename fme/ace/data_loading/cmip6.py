import dataclasses
import logging
import os
from collections.abc import Sequence

import pandas as pd

from fme.core.dataset.concat import ConcatDatasetConfig
from fme.core.dataset.config import DatasetConfigABC
from fme.core.dataset.dataset import DatasetABC
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.xarray import XarrayDataConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Cmip6DataConfig(DatasetConfigABC):
    """Configuration for loading CMIP6 processed datasets.

    Reads index.csv from data_dir, filters to matching datasets, and
    delegates to ConcatDatasetConfig for the actual data loading.
    Each (source_id, experiment, variant_label) combination produces
    one XarrayDataConfig entry.

    Parameters:
        data_dir: Path to the directory containing index.csv and zarr stores.
        source_ids: Source model IDs to include. If None, all available.
        experiments: Experiment IDs to include.
        realizations: Realization numbers (r values) to include. If None, all.
    """

    data_dir: str
    source_ids: list[str] | None = None
    experiments: list[str] = dataclasses.field(
        default_factory=lambda: ["historical", "ssp585"]
    )
    realizations: list[int] | None = None

    def __post_init__(self):
        self._concat_config_cache: ConcatDatasetConfig | None = None

    @property
    def zarr_engine_used(self) -> bool:
        return True

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
        mask &= idx["experiment"].isin(self.experiments)
        if self.realizations is not None:
            mask &= idx["variant_r"].isin(self.realizations)
        filtered = idx[mask]
        if len(filtered) == 0:
            raise ValueError(
                f"No datasets with status='ok' matched the filter criteria "
                f"in {index_path}. "
                f"source_ids={self.source_ids}, "
                f"experiments={self.experiments}, "
                f"realizations={self.realizations}"
            )
        return filtered

    def _build_concat_config(self) -> ConcatDatasetConfig:
        idx = self._load_and_filter_index()
        configs: list[XarrayDataConfig] = []
        for _, row in idx.iterrows():
            data_path = os.path.join(
                self.data_dir,
                row["source_id"],
                row["experiment"],
                row["variant_label"],
            )
            configs.append(
                XarrayDataConfig(
                    data_path=data_path,
                    file_pattern="data.zarr",
                    engine="zarr",
                    labels=[row["label"]],
                )
            )
        logger.info(
            "Cmip6DataConfig: %d datasets from %d source models",
            len(configs),
            idx["source_id"].nunique(),
        )
        return ConcatDatasetConfig(concat=configs, strict=False)

    @property
    def available_labels(self) -> set[str] | None:
        return self._get_concat_config().available_labels

    def build(
        self,
        names: Sequence[str],
        n_timesteps: IntSchedule,
    ) -> tuple[DatasetABC, DatasetProperties]:
        return self._get_concat_config().build(names, n_timesteps)
