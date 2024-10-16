import dataclasses
import datetime
from typing import Sequence

from fme.core.data_loading.requirements import DataRequirements


@dataclasses.dataclass
class CoupledDataRequirements:
    ocean_timestep: datetime.timedelta
    ocean_requirements: DataRequirements
    atmosphere_timestep: datetime.timedelta
    atmosphere_requirements: DataRequirements

    @property
    def timesteps(self) -> Sequence[datetime.timedelta]:
        return [self.ocean_timestep, self.atmosphere_timestep]
