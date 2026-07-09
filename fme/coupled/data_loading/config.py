import dataclasses
from collections.abc import Sequence

from fme.core.dataset.merged import MergeNoConcatDatasetConfig
from fme.core.dataset.utils import accumulate_labels
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.distributed import Distributed


@dataclasses.dataclass
class CoupledDatasetConfig:
    """
    Parameters:
        ocean: Configuration for the ocean dataset.
        atmosphere: Configuration for the atmosphere dataset.
    """

    ocean: XarrayDataConfig | MergeNoConcatDatasetConfig
    atmosphere: XarrayDataConfig | MergeNoConcatDatasetConfig

    @property
    def data_configs(
        self,
    ) -> Sequence[XarrayDataConfig | MergeNoConcatDatasetConfig]:
        return [self.ocean, self.atmosphere]

    @property
    def ice(self) -> None:
        return None

    @property
    def anchor_component_name(self) -> str:
        return "ocean"

    @property
    def coupled_configs(self) -> Sequence["CoupledDatasetConfig"]:
        return [self]


@dataclasses.dataclass
class CoupledIceAtmosphereDatasetConfig:
    """
    Configuration for a coupled dataset with atmosphere and ice components only
    (no ocean).

    Parameters:
        ice: Configuration for the ice dataset.
        atmosphere: Configuration for the atmosphere dataset.
    """

    atmosphere: XarrayDataConfig | MergeNoConcatDatasetConfig
    ice: XarrayDataConfig | MergeNoConcatDatasetConfig

    @property
    def data_configs(
        self,
    ) -> Sequence[XarrayDataConfig | MergeNoConcatDatasetConfig]:
        return [self.atmosphere, self.ice]

    @property
    def ocean(self) -> None:
        return None

    @property
    def anchor_component_name(self) -> str:
        return "ice"

    @property
    def coupled_configs(self) -> Sequence["CoupledIceAtmosphereDatasetConfig"]:
        return [self]


@dataclasses.dataclass
class CoupledIceOceanDatasetConfig:
    """
    Configuration for a coupled dataset with ocean and ice components only
    (no atmosphere).

    Parameters:
        ocean: Configuration for the ocean dataset.
        ice: Configuration for the ice dataset.
    """

    ocean: XarrayDataConfig | MergeNoConcatDatasetConfig
    ice: XarrayDataConfig | MergeNoConcatDatasetConfig

    @property
    def data_configs(
        self,
    ) -> Sequence[XarrayDataConfig | MergeNoConcatDatasetConfig]:
        return [self.ocean, self.ice]

    @property
    def atmosphere(self) -> None:
        return None

    @property
    def anchor_component_name(self) -> str:
        return "ocean"

    @property
    def coupled_configs(self) -> Sequence["CoupledIceOceanDatasetConfig"]:
        return [self]


@dataclasses.dataclass
class CoupledAtmosphereIceOceanDatasetConfig:
    """
    Parameters:
        ocean: Configuration for the ocean dataset.
        ice: Configuration for the ice dataset.
        atmosphere: Configuration for the atmosphere dataset.
    """

    ocean: XarrayDataConfig | MergeNoConcatDatasetConfig
    ice: XarrayDataConfig | MergeNoConcatDatasetConfig
    atmosphere: XarrayDataConfig | MergeNoConcatDatasetConfig

    @property
    def data_configs(
        self,
    ) -> Sequence[XarrayDataConfig | MergeNoConcatDatasetConfig]:
        return [self.ocean, self.ice, self.atmosphere]

    @property
    def anchor_component_name(self) -> str:
        return "ocean"

    @property
    def coupled_configs(self) -> Sequence["CoupledAtmosphereIceOceanDatasetConfig"]:
        return [self]


@dataclasses.dataclass
class CoupledDatasetWithOptionalOceanConfig:
    """
    Config where atmosphere is always present; ocean and ice are optional.
    Ocean is always the anchor: if ocean is None, a dummy will be built
    from the dataset_info during inference.

    Parameters:
        atmosphere: Configuration for the atmosphere dataset.
        ocean: Optional configuration for the ocean dataset.
        ice: Optional configuration for the ice dataset.
    """

    atmosphere: XarrayDataConfig | MergeNoConcatDatasetConfig
    ocean: XarrayDataConfig | MergeNoConcatDatasetConfig | None = None
    ice: XarrayDataConfig | MergeNoConcatDatasetConfig | None = None

    @property
    def data_configs(
        self,
    ) -> Sequence[XarrayDataConfig | MergeNoConcatDatasetConfig | None]:
        return [self.ocean, self.ice, self.atmosphere]

    @property
    def anchor_component_name(self) -> str:
        # Ocean is always the anchor; when ocean is None a dummy is created
        # from dataset_info during inference.
        return "ocean"


def build_coupled_dataset_config(
    atmosphere: XarrayDataConfig | MergeNoConcatDatasetConfig | None = None,
    ice: XarrayDataConfig | MergeNoConcatDatasetConfig | None = None,
    ocean: XarrayDataConfig | MergeNoConcatDatasetConfig | None = None,
) -> (
    CoupledAtmosphereIceOceanDatasetConfig
    | CoupledDatasetConfig
    | CoupledIceAtmosphereDatasetConfig
    | CoupledIceOceanDatasetConfig
    | CoupledDatasetWithOptionalOceanConfig
):
    """
    Returns the appropriate coupled dataset config for the combination of
    components provided.  At least one component must be non-None.

    Coupling priorities (for inference anchor selection):
        ocean > ice > atmosphere
    """
    if atmosphere is not None and ice is not None and ocean is not None:
        return CoupledAtmosphereIceOceanDatasetConfig(
            ocean=ocean, ice=ice, atmosphere=atmosphere
        )
    if atmosphere is not None and ocean is not None:
        return CoupledDatasetConfig(ocean=ocean, atmosphere=atmosphere)
    if atmosphere is not None and ice is not None:
        # ocean is not provided; use CoupledDatasetWithOptionalOceanConfig so that
        # a dummy ocean (driven by dataset_info) is created as the anchor during
        # inference, preserving the ocean-driven batch schedule.
        return CoupledDatasetWithOptionalOceanConfig(
            atmosphere=atmosphere, ice=ice, ocean=None
        )
    if ocean is not None and ice is not None:
        return CoupledIceOceanDatasetConfig(ocean=ocean, ice=ice)
    if atmosphere is not None:
        # atmosphere only: ocean/ice anchor will be provided as a dummy
        return CoupledDatasetWithOptionalOceanConfig(atmosphere=atmosphere, ocean=None)
    raise ValueError("At least one of atmosphere, ice, or ocean must be provided.")


@dataclasses.dataclass
class CoupledConcatDatasetConfig:
    """
    Parameters:
        concat: A sequence of configurations each defining a coupled dataset
            to be loaded. This sequence of datasets will be concatenated.
    """

    concat: Sequence[
        CoupledDatasetConfig
        | CoupledIceOceanDatasetConfig
        | CoupledIceAtmosphereDatasetConfig
        | CoupledAtmosphereIceOceanDatasetConfig
    ]

    @property
    def coupled_configs(
        self,
    ) -> Sequence[
        CoupledDatasetConfig
        | CoupledIceOceanDatasetConfig
        | CoupledIceAtmosphereDatasetConfig
        | CoupledAtmosphereIceOceanDatasetConfig
    ]:
        return self.concat


@dataclasses.dataclass
class CoupledDataLoaderConfig:
    """
    Parameters:
        dataset: A sequence of configurations each defining a coupled dataset
            to be loaded. This sequence of datasets will be concatenated.
        batch_size: Number of samples per batch.
        num_data_workers: Number of parallel workers to use for data loading.
        prefetch_factor: how many batches a single data worker will attempt to
            hold in host memory at a given time.
        strict_ensemble: Whether to enforce that the concatenated ensemble
            members have the same dimensions and coordinates.

    """

    dataset: (
        CoupledConcatDatasetConfig
        | CoupledDatasetConfig
        | CoupledIceOceanDatasetConfig
        | CoupledIceAtmosphereDatasetConfig
        | CoupledAtmosphereIceOceanDatasetConfig
    )
    batch_size: int
    num_data_workers: int = 1
    prefetch_factor: int | None = None
    strict_ensemble: bool = True

    def __post_init__(self):
        dist = Distributed.get_instance()
        if self.batch_size % dist.world_size != 0:
            raise ValueError(
                "batch_size must be divisible by the number of parallel "
                f"workers, got {self.batch_size} and {dist.world_size}"
            )
        self._zarr_engine_used = any(
            ds.zarr_engine_used
            for ds_coupled in self.dataset.coupled_configs
            for ds in ds_coupled.data_configs
            if ds is not None
        )

    @property
    def atmosphere_available_labels(self) -> set[str] | None:
        """
        Return the labels that are available in the atmosphere dataset.
        """
        return accumulate_labels(
            [
                ds.atmosphere.available_labels
                for ds in self.dataset.coupled_configs
                if ds.atmosphere is not None
            ]
        )

    @property
    def ice_available_labels(self) -> set[str] | None:
        """
        Return the labels that are available in the ice dataset.
        """
        return accumulate_labels(
            [
                ds.ice.available_labels
                for ds in self.dataset.coupled_configs
                if ds.ice is not None
            ]
        )

    @property
    def ocean_available_labels(self) -> set[str] | None:
        """
        Return the labels that are available in the ocean dataset.
        """
        return accumulate_labels(
            [
                ds.ocean.available_labels
                for ds in self.dataset.coupled_configs
                if ds.ocean is not None
            ]
        )

    @property
    def zarr_engine_used(self) -> bool:
        """
        Whether the dataset uses the Zarr engine in any of its components.
        """
        return self._zarr_engine_used
