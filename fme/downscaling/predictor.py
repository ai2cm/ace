from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import dacite
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.downscaling.data import StaticInputs, load_coords_from_path
from fme.downscaling.models import DiffusionModelConfig
from fme.downscaling.requirements import DataRequirements

if TYPE_CHECKING:
    from fme.downscaling.predictors.base import BasePredictor


@dataclasses.dataclass
class _CheckpointModelConfigSelector:
    wrapper: DiffusionModelConfig

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> DiffusionModelConfig:
        return dacite.from_dict(
            data={"wrapper": state}, data_class=cls, config=dacite.Config(strict=True)
        ).wrapper


@dataclasses.dataclass
class CheckpointModelConfig:
    """
    This class specifies a diffusion model loaded from a checkpoint file.

    Parameters:
        checkpoint_path: The path to the checkpoint file.
        rename: Optional mapping of {old: new} model input/output names to rename.
        static_inputs: Optional mapping of {field_name: path} for static inputs to
            the model. Useful when no fine res data is available during evaluation
            but the model requires static input data. Raises an error if the
            checkpoint already has static inputs from training.
        fine_topography_path: Deprecated. Use static_inputs instead.
        fine_coordinates_path: Optional path to a netCDF/zarr file containing lat/lon
            coordinates for the full fine domain. Used for old checkpoints that have
            no static_inputs and no stored fine_coords.
        model_updates: Optional mapping of {key: new_value} model config updates to
            apply when loading the model. This is useful for running evaluation with
            updated parameters than at training time. Use with caution; not all
            parameters can or should be updated at evaluation time.
    """

    checkpoint_path: str
    rename: dict[str, str] | None = None
    static_inputs: dict[str, str] | None = None
    fine_topography_path: str | None = None
    fine_coordinates_path: str | None = None
    model_updates: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._checkpoint_is_loaded = False
        self._rename = self.rename or {}
        if "module" in (self.model_updates or {}):
            raise ValueError("'module' cannot be updated in model_updates.")
        if self.fine_topography_path is not None:
            raise ValueError(
                "fine_topography_path is deprecated and will be removed in "
                "a future release. Use static_inputs instead, "
                "e.g., static_inputs: {HGTsfc: <path>}.",
            )

    @property
    def _checkpoint(self) -> Mapping[str, Any]:
        if not self._checkpoint_is_loaded:
            checkpoint_data = torch.load(self.checkpoint_path, weights_only=False)
            checkpoint_data["model"]["config"]["in_names"] = [
                self._rename.get(name, name)
                for name in checkpoint_data["model"]["config"]["in_names"]
            ]
            checkpoint_data["model"]["config"]["out_names"] = [
                self._rename.get(name, name)
                for name in checkpoint_data["model"]["config"]["out_names"]
            ]
            self._checkpoint_data = checkpoint_data
            self._checkpoint_is_loaded = True
            if self.model_updates is not None:
                for k, v in self.model_updates.items():
                    checkpoint_data["model"]["config"][k] = v
        return self._checkpoint_data

    @staticmethod
    def _get_coords_backwards_compatible(
        coords_from_state: dict | None,
        fine_coordinates_path: str | None,
    ) -> LatLonCoordinates:
        if coords_from_state and fine_coordinates_path:
            raise ValueError(
                "Checkpoint contains fine coordinates but fine_coordinates_path is also"
                " provided. Backwards compatibility loading only supports a single "
                "source of fine coordinates info."
            )
        if coords_from_state is not None:
            return LatLonCoordinates(
                lat=coords_from_state["lat"],
                lon=coords_from_state["lon"],
            )
        elif fine_coordinates_path is not None:
            return load_coords_from_path(fine_coordinates_path)
        else:
            raise ValueError(
                "No fine coordinates found in checkpoint state and no "
                "fine_coordinates_path provided. One of these must be provided to "
                "load the model using CheckpointModelConfig."
            )

    def build(self) -> BasePredictor:
        from fme.downscaling.predictors.base import BasePredictor

        checkpoint_model: dict = self._checkpoint["model"]
        full_fine_coords = self._get_coords_backwards_compatible(
            checkpoint_model.get("full_fine_coords"),
            self.fine_coordinates_path,
        )
        static_inputs = StaticInputs.from_state_backwards_compatible(
            state=checkpoint_model.get("static_inputs") or {},
            static_inputs_config=self.static_inputs or {},
        )
        model = _CheckpointModelConfigSelector.from_state(
            self._checkpoint["model"]["config"]
        ).build(
            coarse_shape=self._checkpoint["model"]["coarse_shape"],
            downscale_factor=self._checkpoint["model"]["downscale_factor"],
            full_fine_coords=full_fine_coords,
            rename=self._rename,
            static_inputs=static_inputs,
        )
        model.module.load_state_dict(self._checkpoint["model"]["module"])
        model.module.eval()
        return BasePredictor(model)

    @property
    def data_requirements(self) -> DataRequirements:
        in_names = self.in_names
        out_names = self.out_names
        return DataRequirements(
            fine_names=out_names,
            coarse_names=list(set(in_names).union(out_names)),
            n_timesteps=1,
            use_fine_topography=self._checkpoint["model"]["config"][
                "use_fine_topography"
            ],
        )

    @property
    def in_names(self):
        return self._checkpoint["model"]["config"]["in_names"]

    @property
    def out_names(self):
        return self._checkpoint["model"]["config"]["out_names"]
