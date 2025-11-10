import dataclasses

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.requirements import DataRequirements
from fme.ace.stepper.insolation.config import Insolation, InsolationConfig
from fme.core.dataset_info import DatasetInfo


@dataclasses.dataclass
class DerivedForcingsConfig:
    """Configuration for computing derived forcings.

    Parameters:
        insolation: Optional configuration for computing derived insolation.
    """

    insolation: InsolationConfig | None = None

    def build(self, dataset_info: DatasetInfo) -> "ForcingDeriver":
        """Build a ForcingDeriver insstance with the current configuration.

        Args:
            dataset_info: Dataset information associated with the Stepper.
        """
        if self.insolation is not None:
            timestep = dataset_info.timestep
            horizontal_coordinates = dataset_info.horizontal_coordinates
            insolation_deriver = self.insolation.build(timestep, horizontal_coordinates)
        else:
            insolation_deriver = None
        forcing_deriver = ForcingDeriver(insolation_deriver)
        return forcing_deriver

    def update_requirements(self, requirements: DataRequirements) -> DataRequirements:
        """Add or remove names from the requirements associated with derived forcings.

        Args:
            requirements: The requirements to update.
        """
        if self.insolation is not None:
            requirements = self.insolation.update_requirements(requirements)
        return requirements

    def validate_replacement(self, replacement: "DerivedForcingsConfig") -> None:
        """Check that a replacement configuration is compatible with the current.

        Args:
            replacement: The configuration replacing the current configuration.

        Raises:
            ValueError
                If the ``insolation_name`` of the replacement configuration is
                incompatible with the current.
        """
        if self.insolation is not None and replacement.insolation is not None:
            original_insolation_name = self.insolation.insolation_name
            if original_insolation_name != replacement.insolation.insolation_name:
                raise ValueError(
                    f"Replacement insolation_name should match the original "
                    f"insolation_name ({original_insolation_name!r}). Got "
                    f"{replacement.insolation.insolation_name!r}."
                )


class ForcingDeriver:
    """A class orchestrating computation of derived forcings.

    Parameters:
        insolation: Optional insolation for computing derived insolation.
    """

    def __init__(self, insolation: Insolation | None):
        self.insolation = insolation

    def __call__(self, forcing: BatchData) -> BatchData:
        """Compute the derived forcings.

        Args:
            forcing: Data to compute the derived forcings from.

        Returns:
            Data updated to include derived forcing variables.
        """
        forcing_dict = forcing.data
        if self.insolation is not None:
            forcing_dict = self.insolation.compute(forcing.time, forcing_dict)
        return BatchData(
            forcing_dict,
            forcing.time,
            forcing.labels,
            forcing.horizontal_dims,
            forcing.n_ensemble,
        )
