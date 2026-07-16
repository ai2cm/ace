import dataclasses
import datetime
import os
from collections.abc import Mapping

import cftime
import numpy.typing as npt

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.main import DataWriterConfig, PairedDataWriter
from fme.ace.inference.data_writer.segment import SegmentContext
from fme.core.cloud import makedirs
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.generics.writer import WriterABC
from fme.coupled.data_loading.batch_data import (
    CoupledPairedData,
    CoupledPrognosticState,
)
from fme.coupled.data_loading.data_typing import CoupledCoords

OCEAN_OUTPUT_DIR_NAME = "ocean"
ATMOSPHERE_OUTPUT_DIR_NAME = "atmosphere"


def _component_segment_context(
    segment_context: SegmentContext | None, component_dir_name: str
) -> SegmentContext | None:
    """Point a run-level segment context at one component's subdirectories."""
    if segment_context is None:
        return None
    return dataclasses.replace(
        segment_context,
        run_dir=os.path.join(segment_context.run_dir, component_dir_name),
        segment_dir=os.path.join(segment_context.segment_dir, component_dir_name),
        previous_segment_dir=(
            os.path.join(segment_context.previous_segment_dir, component_dir_name)
            if segment_context.previous_segment_dir is not None
            else None
        ),
    )


@dataclasses.dataclass
class CoupledDataWriterConfig:
    """
    Configuration for coupled inference data writers.

    Parameters:
        ocean: Configuration for ocean data writer.
        atmosphere: Configuration for atmosphere data writer.
    """

    ocean: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )
    atmosphere: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )

    def build_paired(
        self,
        experiment_dir: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        n_timesteps_ocean: int,
        n_timesteps_atmosphere: int,
        ocean_timestep: datetime.timedelta,
        atmosphere_timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: CoupledCoords,
        dataset_metadata: dict[str, DatasetMetadata],
        segment_context: SegmentContext | None = None,
    ) -> "CoupledPairedDataWriter":
        ocean_dir = os.path.join(experiment_dir, OCEAN_OUTPUT_DIR_NAME)
        makedirs(ocean_dir, exist_ok=True)
        atmos_dir = os.path.join(experiment_dir, ATMOSPHERE_OUTPUT_DIR_NAME)
        makedirs(atmos_dir, exist_ok=True)
        return CoupledPairedDataWriter(
            ocean_writer=self.ocean.build_paired(
                experiment_dir=ocean_dir,
                initial_condition_times=initial_condition_times,
                n_timesteps=n_timesteps_ocean,
                timestep=ocean_timestep,
                variable_metadata=variable_metadata,
                coords=coords.ocean,
                dataset_metadata=dataset_metadata["ocean"],
                segment_context=_component_segment_context(
                    segment_context, OCEAN_OUTPUT_DIR_NAME
                ),
            ),
            atmosphere_writer=self.atmosphere.build_paired(
                experiment_dir=atmos_dir,
                initial_condition_times=initial_condition_times,
                n_timesteps=n_timesteps_atmosphere,
                timestep=atmosphere_timestep,
                variable_metadata=variable_metadata,
                coords=coords.atmosphere,
                dataset_metadata=dataset_metadata["atmosphere"],
                segment_context=_component_segment_context(
                    segment_context, ATMOSPHERE_OUTPUT_DIR_NAME
                ),
            ),
        )


class CoupledPairedDataWriter(WriterABC[CoupledPrognosticState, CoupledPairedData]):
    def __init__(
        self,
        ocean_writer: PairedDataWriter,
        atmosphere_writer: PairedDataWriter,
    ):
        self._ocean_writer = ocean_writer
        self._atmosphere_writer = atmosphere_writer

    def write(self, data: CoupledPrognosticState, filename: str):
        self._ocean_writer.write(data.ocean_data, filename)
        self._atmosphere_writer.write(data.atmosphere_data, filename)

    def append_batch(self, batch: CoupledPairedData):
        self._ocean_writer.append_batch(batch.ocean_data)
        self._atmosphere_writer.append_batch(batch.atmosphere_data)

    def flush(self):
        self._ocean_writer.flush()
        self._atmosphere_writer.flush()

    def finalize(self):
        self._ocean_writer.finalize()
        self._atmosphere_writer.finalize()
