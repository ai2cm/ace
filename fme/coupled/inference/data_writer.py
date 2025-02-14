import dataclasses
import datetime
import os
from typing import Iterable, Mapping, Optional

from fme.ace.inference.data_writer.main import DataWriterConfig, PairedDataWriter
from fme.ace.inference.loop import write_reduced_metrics as write_reduced_metrics_
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.generics.writer import WriterABC
from fme.coupled.aggregator import InferenceEvaluatorAggregator
from fme.coupled.data_loading.batch_data import (
    CoupledPairedData,
    CoupledPrognosticState,
)
from fme.coupled.data_loading.data_typing import CoupledCoords

OCEAN_OUTPUT_DIR_NAME = "ocean"
ATMOSPHERE_OUTPUT_DIR_NAME = "atmosphere"


def write_reduced_metrics(
    aggregator: InferenceEvaluatorAggregator,
    data_coords: CoupledCoords,
    experiment_dir: str,
    excluded: Optional[Iterable[str]] = None,
):
    """
    Write the reduced metrics to disk. Each component aggregator will write
    netCDFs to their corresponding sub-directories.

    Args:
        aggregator: The aggregator to write metrics from.
        data_coords: Coordinates to assign to the datasets.
        experiment_dir: Base path to write the metrics to.
        excluded: Names of metrics to exclude from writing.

    """
    ocean_dir = os.path.join(experiment_dir, OCEAN_OUTPUT_DIR_NAME)
    atmosphere_dir = os.path.join(experiment_dir, ATMOSPHERE_OUTPUT_DIR_NAME)
    if not os.path.exists(ocean_dir):
        os.makedirs(ocean_dir)
    if not os.path.exists(atmosphere_dir):
        os.makedirs(atmosphere_dir)
    write_reduced_metrics_(
        aggregator.ocean,
        data_coords.ocean,
        ocean_dir,
        excluded=excluded,
    )
    write_reduced_metrics_(
        aggregator.atmosphere,
        data_coords.atmosphere,
        atmosphere_dir,
        excluded=excluded,
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
        n_initial_conditions: int,
        n_timesteps_ocean: int,
        n_timesteps_atmosphere: int,
        ocean_timestep: datetime.timedelta,
        atmosphere_timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: CoupledCoords,
    ) -> "CoupledPairedDataWriter":
        ocean_dir = os.path.join(experiment_dir, OCEAN_OUTPUT_DIR_NAME)
        if not os.path.exists(ocean_dir):
            os.makedirs(ocean_dir)
        atmos_dir = os.path.join(experiment_dir, ATMOSPHERE_OUTPUT_DIR_NAME)
        if not os.path.exists(atmos_dir):
            os.makedirs(atmos_dir)
        return CoupledPairedDataWriter(
            ocean_writer=self.ocean.build_paired(
                experiment_dir=ocean_dir,
                n_initial_conditions=n_initial_conditions,
                n_timesteps=n_timesteps_ocean,
                timestep=ocean_timestep,
                variable_metadata=variable_metadata,
                coords=coords.ocean,
            ),
            atmosphere_writer=self.atmosphere.build_paired(
                experiment_dir=atmos_dir,
                n_initial_conditions=n_initial_conditions,
                n_timesteps=n_timesteps_atmosphere,
                timestep=atmosphere_timestep,
                variable_metadata=variable_metadata,
                coords=coords.atmosphere,
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
