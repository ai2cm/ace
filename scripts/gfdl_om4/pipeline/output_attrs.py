"""Output labels that the regridding changes: rotated vector components and
full-cell / per-ocean-area twins. Kept free of beam so it can be tested
without the pipeline's runtime dependencies."""

import re

EASTWARD, NORTHWARD = "eastward", "northward"

# CF standard_names of grid-relative components -> geographic counterparts.
# Any other standard_name on a rotated component is dropped.
GEOGRAPHIC_STANDARD_NAMES = {
    "sea_water_x_velocity": "eastward_sea_water_velocity",
    "sea_water_y_velocity": "northward_sea_water_velocity",
    "surface_downward_x_stress": "surface_downward_eastward_stress",
    "surface_downward_y_stress": "surface_downward_northward_stress",
    "sea_ice_x_velocity": "eastward_sea_ice_velocity",
    "sea_ice_y_velocity": "northward_sea_ice_velocity",
}

CELL_AREA_BASIS = "fraction of cell area"
OCEAN_AREA_BASIS = "fraction of ocean area"


def geographic_component_attrs(attrs: dict, name: str, direction: str) -> dict:
    """``attrs`` of grid-relative component ``name`` relabeled for its rotated
    ``direction`` (eastward from x, northward from y) component.

    The axis word in ``long_name`` (a standalone x or y) becomes
    ``direction``; without one, ``direction`` is appended.
    """
    axis = {EASTWARD: "x", NORTHWARD: "y"}[direction]
    out = dict(attrs)
    long_name = attrs.get("long_name", name)
    relabeled, count = re.subn(
        rf"\b{axis}\b",
        lambda m: direction.capitalize() if m.group().isupper() else direction,
        long_name,
        flags=re.IGNORECASE,
    )
    out["long_name"] = relabeled if count else f"{long_name}, {direction} component"
    standard_name = out.pop("standard_name", None)
    if standard_name in GEOGRAPHIC_STANDARD_NAMES:
        out["standard_name"] = GEOGRAPHIC_STANDARD_NAMES[standard_name]
    return out


def area_basis_long_name(attrs: dict, name: str, basis: str) -> str:
    """Long name of a fraction twin: the source long_name and its area basis
    (CELL_AREA_BASIS for the full-cell regrid, OCEAN_AREA_BASIS for the
    per-ocean-area one)."""
    return f"{attrs.get('long_name', name)}, {basis}"
