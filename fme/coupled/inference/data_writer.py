import dataclasses
import datetime
import os
from collections.abc import Mapping

import cftime
import numpy.typing as npt

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.main import DataWriterConfig, PairedDataWriter
from fme.core.cloud import makedirs
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.generics.writer import WriterABC
from fme.coupled.data_loading.batch_data import (
    CoupledPairedData,
    CoupledPrognosticState,
)
from fme.coupled.data_loading.data_typing import CoupledCoords

OCEAN_OUTPUT_DIR_NAME = "ocean"
ICE_OUTPUT_DIR_NAME = "ice"
ATMOSPHERE_OUTPUT_DIR_NAME = "atmosphere"


@dataclasses.dataclass
class CoupledDataWriterConfig:
    """
    Configuration for coupled inference data writers.

    Parameters:
        ocean: Configuration for ocean data writer.
        ice: Configuration for ice data writer.
        atmosphere: Configuration for atmosphere data writer.
    """

    ocean: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )
    ice: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )
    atmosphere: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )

    def build_paired(
        self,
        experiment_dir: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        variable_metadata: Mapping[str, VariableMetadata],
        coords: CoupledCoords,
        dataset_metadata: dict[str, DatasetMetadata],
        n_timesteps_ocean: int | None = None,
        n_timesteps_ice: int | None = None,
        n_timesteps_atmosphere: int | None = None,
        ocean_timestep: datetime.timedelta | None = None,
        ice_timestep: datetime.timedelta | None = None,
        atmosphere_timestep: datetime.timedelta | None = None,
    ) -> "CoupledPairedDataWriter":
        ocean_writer = None
        ice_writer = None
        atmosphere_writer = None
        if ocean_timestep is not None:
            ocean_dir = os.path.join(experiment_dir, OCEAN_OUTPUT_DIR_NAME)
            makedirs(ocean_dir, exist_ok=True)
            ocean_writer=self.ocean.build_paired(
                experiment_dir=ocean_dir,
                initial_condition_times=initial_condition_times,
                n_timesteps=n_timesteps_ocean,
                timestep=ocean_timestep,
                variable_metadata=variable_metadata,
                coords=coords.ocean,
                dataset_metadata=dataset_metadata["ocean"],
            )
        if ice_timestep is not None:
            ice_dir = os.path.join(experiment_dir, ICE_OUTPUT_DIR_NAME)
            makedirs(ice_dir, exist_ok=True)
            ice_writer=self.ice.build_paired(
                experiment_dir=ice_dir,
                initial_condition_times=initial_condition_times,
                n_timesteps=n_timesteps_ice,
                timestep=ice_timestep,
                variable_metadata=variable_metadata,
                coords=coords.ice,
                dataset_metadata=dataset_metadata["ice"],
            )
        if atmosphere_timestep is not None:
            atmos_dir = os.path.join(experiment_dir, ATMOSPHERE_OUTPUT_DIR_NAME)
            makedirs(atmos_dir, exist_ok=True)
            atmosphere_writer=self.atmosphere.build_paired(
                experiment_dir=atmos_dir,
                initial_condition_times=initial_condition_times,
                n_timesteps=n_timesteps_atmosphere,
                timestep=atmosphere_timestep,
                variable_metadata=variable_metadata,
                coords=coords.atmosphere,
                dataset_metadata=dataset_metadata["atmosphere"],
            )
        
        return CoupledPairedDataWriter(
            ocean_writer=ocean_writer,
            ice_writer=ice_writer,
            atmosphere_writer=atmosphere_writer
        )


class CoupledPairedDataWriter(WriterABC[CoupledPrognosticState, CoupledPairedData]):
    def __init__(
        self,
        ocean_writer: PairedDataWriter | None = None,
        ice_writer: PairedDataWriter | None = None,
        atmosphere_writer: PairedDataWriter | None = None,
    ):
        self._ocean_writer = ocean_writer
        self._ice_writer = ice_writer
        self._atmosphere_writer = atmosphere_writer

    def write(self, data: CoupledPrognosticState, filename: str):
        if self._ocean_writer is not None:
            self._ocean_writer.write(data.ocean_data, filename)
        if self._ice_writer is not None:
            self._ice_writer.write(data.ice_data, filename)
        if self._atmosphere_writer is not None:
            self._atmosphere_writer.write(data.atmosphere_data, filename)

    def append_batch(self, batch: CoupledPairedData):
        if self._ocean_writer is not None:
            self._ocean_writer.append_batch(batch.ocean_data)
        if self._ice_writer is not None:
            self._ice_writer.append_batch(batch.ice_data)
        if self._atmosphere_writer is not None:
            self._atmosphere_writer.append_batch(batch.atmosphere_data)

    def flush(self):
        if self._ocean_writer is not None:
            self._ocean_writer.flush()
        if self._ice_writer is not None:
            self._ice_writer.flush()
        if self._atmosphere_writer is not None:
            self._atmosphere_writer.flush()

    def finalize(self):
        if self._ocean_writer is not None:
            self._ocean_writer.finalize()
        if self._ice_writer is not None:
            self._ice_writer.finalize()
        if self._atmosphere_writer is not None:
            self._atmosphere_writer.finalize()
