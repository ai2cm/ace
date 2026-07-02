"""Schema version stamping for CMIP6 archive datasets.

Every per-dataset ``metadata.json`` sidecar records the
``SCHEMA_VERSION`` that was current when the dataset was written.
``migrate.py`` reads each sidecar, finds the registered migrations
between the recorded version and the current ``SCHEMA_VERSION``, and
applies them in order. Datasets predating this framework — sidecars
without a ``schema_version`` field — are treated as version
``"0.0.0"`` so the chain still applies cleanly.

The version is a hand-maintained semver string. Bump it (and add a
matching migration to ``migrations/``) whenever the on-disk schema
changes — variable renames, unit rescales, new derived channels, mask
convention changes, etc.

**Clean break at 1.0.0.** The ``0.x`` line (the Pangeo-sourced multi-model
cohort, migrations up to ``0.9.0``) and the ``1.0.0`` line are intentionally
NOT connected by a migration: ``1.0.0`` is the ESGF-sourced, per-field
time-varying-mask + real-``ta`` + layer-thickness schema, which cannot be
derived from ``0.9.0`` data (the new fields require re-ingesting from ESGF).
There is deliberately no ``0.9.0``→``1.0.0`` migration; ``0.x`` datasets stay
on the ``0.x`` line and ``1.x`` datasets are written fresh. Future ``1.x``
migrations chain from ``1.0.0`` onward.
"""

from dataclasses import dataclass
from typing import Callable

SCHEMA_VERSION: str = "1.0.0"

# Base of the post-clean-break ``1.x`` migration line. Datasets at or above this
# version migrate among themselves toward ``SCHEMA_VERSION``; there is no
# migration path from the ``0.x`` line across this boundary (see module docstring).
CLEAN_BREAK_BASE_VERSION: str = "1.0.0"


@dataclass(frozen=True)
class Migration:
    """One step of the schema migration chain.

    Attributes:
        from_version: Schema version this migration starts from
            (matches the sidecar's ``schema_version`` field).
        to_version: Schema version after this migration runs.
        description: One-line human-readable summary, surfaced in the
            ``migrate.py`` log so dry-runs are informative.
        apply: ``(zarr_path, sidecar_dict) -> updated_sidecar_dict``.
            The function mutates the zarr in place (typically via
            ``xr.open_zarr`` + ``.to_zarr(mode="w")``) and returns the
            new sidecar contents (``schema_version`` bumped, perhaps
            ``variables_present`` updated, plus any free-form notes the
            migration wants to record).
    """

    from_version: str
    to_version: str
    description: str
    apply: Callable[[str, dict], dict]


def parse_version(s: str) -> tuple[int, int, int]:
    """Parse a semver string ``"a.b.c"`` to ``(a, b, c)``.

    Empty / missing strings are treated as ``"0.0.0"`` — the
    convention used for sidecars written before this framework
    landed.
    """
    if not s:
        return (0, 0, 0)
    parts = s.strip().split(".")
    if len(parts) != 3:
        raise ValueError(f"invalid schema version {s!r}: expected 'a.b.c'")
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError as e:
        raise ValueError(f"invalid schema version {s!r}: {e}") from e


def version_lt(a: str, b: str) -> bool:
    """Return True iff ``a`` is strictly older than ``b``."""
    return parse_version(a) < parse_version(b)
