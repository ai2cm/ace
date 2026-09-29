"""Variables derived in the loader from stored fields of a merged dataset.

A requested derived name is swapped for its stored inputs when the names are
split among the merged stores (``expand_names``), and computed on each merged
sample window (``apply``). Inputs that were not requested are then dropped.
The budget terms come from ``fme.core.frozen_mass_budget``, which the ocean
corrector also uses. The names in ``DERIVED_MASK_SOURCE`` get a
``mask_<name>`` in the merged properties (``add_masks``) and are NaN where it
is 0.
"""

import datetime
from collections.abc import Collection, Mapping, Sequence

import torch

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.properties import DatasetProperties
from fme.core.frozen_mass_budget import (
    calving_residue,
    frozen_mass,
    target_frozen_mass_energy_budget_residual,
)
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.typing_ import TensorDict, TensorMapping

_FROZEN_MASS_INPUTS = ["simass", "sisnmass", "sea_surface_fraction"]
_CALVING_RESIDUE_INPUTS = ["hflso", "evs", "prsn"]
_RESIDUAL_INPUTS = (
    _FROZEN_MASS_INPUTS
    + _CALVING_RESIDUE_INPUTS
    + [
        "hfrunoffds",
        "hfds_total_area",
        "sst",
        "land_fraction",
        "ocean_sea_ice_fraction",
        "DSWRFsfc",
        "USWRFsfc",
        "DLWRFsfc",
        "ULWRFsfc",
        "LHTFLsfc",
        "SHTFLsfc",
        "PRATEsfc",
        "total_frozen_precipitation_rate",
    ]
)

DERIVED_INPUTS: dict[str, list[str]] = {
    "frozen_mass": _FROZEN_MASS_INPUTS,
    "calving_residue": _CALVING_RESIDUE_INPUTS,
    "frozen_mass_energy_budget_residual": _RESIDUAL_INPUTS,
}

# The stored inputs that are NaN off the ocean (``mask_2d == 0``). A derived
# name is NaN wherever one of these is, as the stored fields are, so the loss
# masks it there. ``ocean_sea_ice_fraction`` is left out: the stores also hold
# NaN in it on ice-free ocean.
_FROZEN_MASS_NAN_INPUTS = ["simass", "sisnmass"]
_CALVING_RESIDUE_NAN_INPUTS = ["hflso", "evs", "prsn"]
_NAN_INPUTS: dict[str, list[str]] = {
    "frozen_mass": _FROZEN_MASS_NAN_INPUTS,
    "calving_residue": _CALVING_RESIDUE_NAN_INPUTS,
    "frozen_mass_energy_budget_residual": _FROZEN_MASS_NAN_INPUTS
    + _CALVING_RESIDUE_NAN_INPUTS
    + ["hfrunoffds", "hfds_total_area", "sst"],
}

# Each derived name here takes ``mask_<name>`` equal to the stored mask named,
# as ``sea_ice_fraction`` is masked by ``mask_sea_ice_fraction``; a store's
# own ``mask_<name>`` takes precedence.
DERIVED_MASK_SOURCE: dict[str, str] = {
    "frozen_mass": "mask_sea_ice_fraction",
    "frozen_mass_energy_budget_residual": "mask_sea_ice_fraction",
}

DERIVED_METADATA: dict[str, VariableMetadata] = {
    "frozen_mass": VariableMetadata(
        units="kg/m**2", long_name="sea ice plus snow mass per total cell area"
    ),
    "calving_residue": VariableMetadata(
        units="W/m**2", long_name="-hflso + L_v evs - L_f prsn per ocean area"
    ),
    "frozen_mass_energy_budget_residual": VariableMetadata(
        units="W/m**2",
        long_name="frozen mass energy budget residual per total cell area",
    ),
}


def expand_names(names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split requested names into the stored names to load (the requested
    stored names plus the inputs of the derived ones) and the derived names.
    """
    derived = [n for n in names if n in DERIVED_INPUTS]
    stored = [n for n in names if n not in DERIVED_INPUTS]
    for name in derived:
        for input_name in DERIVED_INPUTS[name]:
            if input_name not in stored:
                stored.append(input_name)
    return stored, derived


def apply(
    tensors: TensorDict,
    derived: Collection[str],
    timestep: datetime.timedelta | None,
    keep: Collection[str],
    masks: TensorMapping | None = None,
) -> TensorDict:
    """Add the ``derived`` names to one sample window (time on dim 0) and
    return only the names in ``keep``. Each derived name is NaN where its
    stored ocean inputs are NaN at the same time (land), and at every time
    where its ``mask_<name>`` in ``masks`` (from ``get_masks``) is 0.
    """
    if len(derived) == 0:
        return tensors
    missing = sorted(
        {i for n in derived for i in DERIVED_INPUTS[n]}.difference(tensors)
    )
    if missing:
        raise KeyError(
            f"Derived variables {sorted(derived)} need stored variables "
            f"{missing}, which the merged datasets did not provide."
        )
    out = dict(tensors)
    if "frozen_mass" in derived:
        out["frozen_mass"] = frozen_mass(
            tensors["simass"], tensors["sisnmass"], tensors["sea_surface_fraction"]
        )
    if "calving_residue" in derived:
        out["calving_residue"] = calving_residue(
            tensors["hflso"], tensors["evs"], tensors["prsn"]
        )
    if "frozen_mass_energy_budget_residual" in derived:
        if timestep is None:
            raise ValueError(
                "frozen_mass_energy_budget_residual needs a dataset timestep."
            )
        out["frozen_mass_energy_budget_residual"] = (
            target_frozen_mass_energy_budget_residual(tensors, timestep.total_seconds())
        )
    for name in derived:
        out[name] = _nan_where_inputs_nan(out[name], tensors, _NAN_INPUTS[name])
        mask = (masks or {}).get(f"mask_{name}")
        if mask is not None:
            out[name] = torch.where(
                mask == 0, torch.full_like(out[name], float("nan")), out[name]
            )
    return {k: v for k, v in out.items() if k in keep}


def _nan_where_inputs_nan(
    x: torch.Tensor, tensors: TensorDict, input_names: Sequence[str]
) -> torch.Tensor:
    nan = torch.zeros_like(x, dtype=torch.bool)
    for name in input_names:
        nan = nan | torch.isnan(tensors[name])
    return torch.where(nan, torch.full_like(x, float("nan")), x)


def add_metadata(properties: DatasetProperties, derived: Collection[str]) -> None:
    """Add the ``VariableMetadata`` of the ``derived`` names in place."""
    for name in derived:
        properties.variable_metadata[name] = DERIVED_METADATA[name]


def get_masks(
    masks: Mapping[str, torch.Tensor], derived: Collection[str]
) -> dict[str, torch.Tensor]:
    """Return ``mask_<name>`` for each ``derived`` name in
    ``DERIVED_MASK_SOURCE``: the stored ``mask_<name>`` if ``masks`` holds it,
    else the stored mask it names.

    Raises:
        ValueError: if neither is in ``masks``.
    """
    out: dict[str, torch.Tensor] = {}
    for name in derived:
        if name not in DERIVED_MASK_SOURCE:
            continue
        key, source = f"mask_{name}", DERIVED_MASK_SOURCE[name]
        if key in masks:
            out[key] = masks[key]
        elif source in masks:
            out[key] = masks[source]
        else:
            raise ValueError(
                f"Derived variable {name} is masked like {source}, which "
                f"none of the merged datasets provide (masks: {sorted(masks)})."
            )
    return out


def add_masks(properties: DatasetProperties, derived: Collection[str]) -> None:
    """Replace ``properties.spatial_mask_provider`` with one that also holds
    the ``get_masks`` masks of the ``derived`` names. The provider is replaced,
    not updated, as it may be shared with a merged sub-dataset.
    """
    masks = properties.spatial_mask_provider.masks
    new = {k: v for k, v in get_masks(masks, derived).items() if k not in masks}
    if new:
        properties.spatial_mask_provider = SpatialMaskProvider({**masks, **new})
