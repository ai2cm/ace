import copy
import dataclasses

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.requirements import DataRequirements
from fme.ace.stepper.insolation.config import Insolation, InsolationConfig
from fme.core.atmosphere_data import AtmosphereData
from fme.core.dataset_info import DatasetInfo
from fme.core.typing_ import TensorMapping

DEFAULT_NET_SURFACE_ENERGY_FLUX_NAME = "net_surface_energy_flux"


@dataclasses.dataclass
class NetSurfaceEnergyFluxConfig:
    """Configuration for deriving the net surface energy flux as a forcing.

    The net flux is computed from the constituent atmosphere surface energy
    fluxes via ``AtmosphereData.net_surface_energy_flux`` (the six radiative and
    turbulent fluxes plus the frozen-precipitation latent term), reused verbatim
    with no masking or vertical coordinate.

    Parameters:
        name: name to assign the derived net surface energy flux; must be present
            as an input to your model.
        flux_names: names of the constituent flux variables to load from disk so
            the net flux can be computed. These are added to the data requirements
            in place of ``name``. They must be names that ``AtmosphereData``
            recognizes (e.g. ``DLWRFsfc``, ``LHTFLsfc``, ...).
    """

    name: str = DEFAULT_NET_SURFACE_ENERGY_FLUX_NAME
    flux_names: list[str] = dataclasses.field(default_factory=list)

    def build(self) -> "NetSurfaceEnergyFlux":
        """Build a NetSurfaceEnergyFlux instance with the current configuration."""
        return NetSurfaceEnergyFlux(self)

    def update_requirements(self, requirements: DataRequirements) -> DataRequirements:
        """Add or remove names from the requirements associated with the net flux.

        Removes the derived ``name`` from the load-list and adds the constituent
        flux fields.

        Args:
            requirements: The requirements to update.
        """
        names = copy.deepcopy(requirements.names)
        if self.name in names:
            names.remove(self.name)
            for flux_name in self.flux_names:
                if flux_name not in names:
                    names.append(flux_name)
        return DataRequirements(
            names=names,
            n_timesteps=requirements.n_timesteps,
            allow_missing_variables=requirements.allow_missing_variables,
        )


class NetSurfaceEnergyFlux:
    """A class orchestrating computation of the net surface energy flux.

    Parameters:
        config: Configuration for computing the net surface energy flux.
    """

    def __init__(self, config: NetSurfaceEnergyFluxConfig):
        self.config = config

    def compute(self, tensors: TensorMapping) -> TensorMapping:
        """Compute the net surface energy flux.

        Args:
            tensors: Dictionary of tensors to update; must contain the
                constituent flux fields.

        Returns:
            The tensor dictionary updated to include the net surface energy flux.
        """
        tensors = dict(tensors)  # Shallow copy to avoid mutating input.
        tensors[self.config.name] = AtmosphereData(tensors).net_surface_energy_flux
        return tensors


@dataclasses.dataclass
class DerivedForcingsConfig:
    """Configuration for computing derived forcings.

    Parameters:
        insolation: Optional configuration for computing derived insolation.
        net_surface_energy_flux: Optional configuration for computing the derived
            net surface energy flux.
    """

    insolation: InsolationConfig | None = None
    net_surface_energy_flux: NetSurfaceEnergyFluxConfig | None = None

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
        if self.net_surface_energy_flux is not None:
            net_surface_energy_flux_deriver = self.net_surface_energy_flux.build()
        else:
            net_surface_energy_flux_deriver = None
        forcing_deriver = ForcingDeriver(
            insolation_deriver, net_surface_energy_flux_deriver
        )
        return forcing_deriver

    def update_requirements(self, requirements: DataRequirements) -> DataRequirements:
        """Add or remove names from the requirements associated with derived forcings.

        Args:
            requirements: The requirements to update.
        """
        if self.insolation is not None:
            requirements = self.insolation.update_requirements(requirements)
        if self.net_surface_energy_flux is not None:
            requirements = self.net_surface_energy_flux.update_requirements(
                requirements
            )
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
        if (
            self.net_surface_energy_flux is not None
            and replacement.net_surface_energy_flux is not None
        ):
            original_name = self.net_surface_energy_flux.name
            if original_name != replacement.net_surface_energy_flux.name:
                raise ValueError(
                    f"Replacement net_surface_energy_flux name should match the "
                    f"original name ({original_name!r}). Got "
                    f"{replacement.net_surface_energy_flux.name!r}."
                )


class ForcingDeriver:
    """A class orchestrating computation of derived forcings.

    Parameters:
        insolation: Optional insolation for computing derived insolation.
        net_surface_energy_flux: Optional deriver for the net surface energy flux.
    """

    def __init__(
        self,
        insolation: Insolation | None,
        net_surface_energy_flux: NetSurfaceEnergyFlux | None = None,
    ):
        self.insolation = insolation
        self.net_surface_energy_flux = net_surface_energy_flux

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
        if self.net_surface_energy_flux is not None:
            forcing_dict = self.net_surface_energy_flux.compute(forcing_dict)
        return BatchData(
            data=forcing_dict,
            time=forcing.time,
            labels=forcing.labels,
            horizontal_dims=forcing.horizontal_dims,
            epoch=forcing.epoch,
            n_ensemble=forcing.n_ensemble,
            data_mask=forcing.data_mask,
        )
