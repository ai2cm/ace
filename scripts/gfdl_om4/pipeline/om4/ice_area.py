"""Ice-area weighted regridding of per-ice-area SIS2 fields (rule R1).

SIS2 writes fields such as ``HI``, ``TS`` and ``UI`` per ice area: a mean
over the ice-covered part of a native cell, 0 where there is no ice. A
per-ocean-area regrid of such a field dilutes it with ice-free ocean. Rule R1
weights it by the ice-area fraction ``c`` at the same time:

    X_J = mean_ocean_J(c X) / a_J     where a_J > 0
        = 0                           where a_J = 0, f_J > 0   (ice-free ocean)
        = NaN                         where f_J = 0            (land)

    mean_ocean_J(x) = Σ_i O_Ji m_i x_i / (A_J f_J)    (regrid_normalized)
    a_J = mean_ocean_J(c)

``i``: native cell; ``J``: target cell; ``f_J``: regridded ocean fraction.
The products ``c X`` are formed on the native grid and regridded per ocean
area like any other field, so the per-ocean-area identity
(pipeline.native_area.check_field_identities) checks them before the
division. Kept free of beam so it can be tested without the pipeline's
runtime dependencies.
"""

import numpy as np
import xarray as xr

from ..ocean_emulators_port import OCEAN_FRACTION_THRESHOLD
from ..postprocess import append_derivation
from .config import StreamConfig


def product_name(name: str) -> str:
    """Chunk-private name of the native product ``c · name``."""
    return f"{name}__ice_area_product"


def add_ice_area_products(ds: xr.Dataset, stream: StreamConfig) -> xr.Dataset:
    """Add ``c X`` under :func:`product_name` for each ice-area weighted ``X``
    in ``ds``; ``ds[X]`` is left as is."""
    weight = stream.ice_area_weight
    names = [name for name in stream.ice_area_weighted if name in ds.data_vars]
    if names and weight not in ds.data_vars:
        raise AssertionError(
            f"stream {stream.name!r}: ice_area_weight {weight!r} missing from "
            f"the chunk holding {names}"
        )
    ds = ds.copy()
    for name in names:
        product = ds[name].astype(np.float64) * ds[weight].astype(np.float64)
        ds[product_name(name)] = product.assign_attrs(ds[name].attrs)
    return ds


def _weight_note(stream: StreamConfig) -> str:
    note = f"{stream.ice_area_weight}"
    if stream.ice_area_weight_store is not None:
        note += f" (from {stream.ice_area_weight_store})"
    return note


CELL_METHODS_ATTR = "cell_methods"
# CF area methods: R1 values are means over the ice-covered part of each
# cell; the named products are means over its ocean part.
ICE_AREA_METHOD = "area: mean where sea_ice"
OCEAN_AREA_METHOD = "area: mean where sea"


def _with_area_method(attrs: dict, method: str) -> dict:
    """``attrs`` with ``method`` appended to its ``cell_methods``."""
    existing = attrs.get(CELL_METHODS_ATTR)
    return {
        **attrs,
        CELL_METHODS_ATTR: f"{existing} {method}" if existing else method,
    }


def ice_area_weighted_means(
    regridded: xr.Dataset, stream: StreamConfig, ocean_fraction: xr.DataArray
) -> tuple[xr.Dataset, xr.Dataset]:
    """Rule R1 on a per-ocean-area regrid holding the products.

    Returns ``regridded`` with each ice-area weighted ``X`` replaced by its
    R1 quotient and every product removed, and the named products
    (``stream.ice_area_products``: output name -> ``mean_ocean_J(c X)``).
    """
    weight = stream.ice_area_weight
    names = [
        name for name in stream.ice_area_weighted if product_name(name) in regridded
    ]
    if not names:
        return regridded, xr.Dataset()
    a = regridded[weight]
    ice = a > 0
    ocean = ocean_fraction > OCEAN_FRACTION_THRESHOLD
    weight_note = _weight_note(stream)
    out = regridded.drop_vars([product_name(name) for name in names])
    products = xr.Dataset()
    for name in names:
        num = regridded[product_name(name)]
        quotient = xr.where(ice, num / a.where(ice), 0.0).where(ocean)
        quotient.attrs = _with_area_method(regridded[name].attrs, ICE_AREA_METHOD)
        out[name] = append_derivation(
            quotient,
            f"ice-area weighted: ocean-area mean of {name} x {weight_note} "
            f"divided by the ocean-area mean of {stream.ice_area_weight}; 0 where "
            "that is 0 over ocean, NaN over land",
        )
        if name in stream.ice_area_products:
            product = num.copy()
            attrs = regridded[name].attrs
            product.attrs = _with_area_method(
                {
                    "long_name": (
                        f"{attrs.get('long_name', name)} times "
                        f"{stream.ice_area_weight}, mean over ocean area"
                    ),
                    **({"units": attrs["units"]} if "units" in attrs else {}),
                    **(
                        {CELL_METHODS_ATTR: attrs[CELL_METHODS_ATTR]}
                        if CELL_METHODS_ATTR in attrs
                        else {}
                    ),
                },
                OCEAN_AREA_METHOD,
            )
            products[stream.ice_area_products[name]] = append_derivation(
                product, f"ocean-area mean of {name} x {weight_note}"
            )
    return out, products
