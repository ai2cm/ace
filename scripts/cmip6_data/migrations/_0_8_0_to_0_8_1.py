"""Migration 0.8.0 → 0.8.1: backfill ``esgf_failed_augment_variables``.

The augment loop in ``process_esgf.py`` catches per-variable failures
so a single bad variable doesn't sink the whole augment pass — but
0.8.0 recorded the failure only as a free-form string in
``warnings``, which the augmentable-selection logic doesn't consult.
A follow-up augment pass would walk per-variable selection, see the
variable was missing from ``variables_present``, and re-attempt it —
re-downloading the same ~1-2 GB to refail the same deterministic
ESMF rc=506 (or HTTP 422 query, etc.). The 0.8.1 schema persists
the prior failures in a dedicated list that the selection step skips
by default; ``--retry-failed-augments`` clears the list to force a
fresh attempt after the upstream issue has been fixed.

This migration backfills the new field from the existing
``warnings`` so the v2 cohort gets the skip behavior without a fresh
augment pass first. No zarr touch — sidecar-only.

Extraction:

- Match free-text warnings of the form ``augment <name> failed: ...``
  (surface-and-ocean) and ``augment day <var> failed: ...``
  (day-cadence).
- For day-cadence, translate the CMIP6 variable name to the canonical
  output name via ``CMIP_TO_OUTPUT_RENAMES`` — that's the convention
  ``esgf_failed_augment_variables`` uses.
- Subtract anything in ``variables_present``: a variable that failed
  in an early attempt but later succeeded (e.g. on a retried pass)
  ends up in the zarr and should *not* be marked as failed.

Idempotent: re-running the migration recomputes the list from
``warnings`` each time. Forward-compatible: post-augment runs that
write to ``esgf_failed_augment_variables`` directly continue to work
because the field's lifecycle is owned by the augment loop, not by
this migration.
"""

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CMIP_TO_OUTPUT_RENAMES  # noqa: E402
from schema_version import Migration  # noqa: E402

# Surface-and-ocean failures use the output name directly:
#   ``augment <output_name> failed: <ExcType>: <msg>``
_RE_AUG_SURFACE = re.compile(r"^augment ([A-Za-z0-9_]+) failed:")
# Day-cadence failures use the CMIP6 variable name:
#   ``augment day <var> failed: <ExcType>: <msg>``
_RE_AUG_DAY = re.compile(r"^augment day ([A-Za-z0-9_]+) failed:")


def _extract_failed_augment_vars(warnings: list[str], present: set[str]) -> list[str]:
    """Return sorted output names a prior augment pass tried and failed.

    Surface-and-ocean and day-cadence warnings use slightly different
    wording — the day-cadence form uses the CMIP6 variable name and
    needs ``CMIP_TO_OUTPUT_RENAMES`` to land on the canonical output
    name that ``variables_present`` /
    ``esgf_failed_augment_variables`` use. Variables that ended up in
    ``present`` are filtered out: a later retry succeeded, so the
    early failure is no longer load-bearing.
    """
    attempted: set[str] = set()
    for w in warnings:
        if not isinstance(w, str):
            continue
        # Order matters: the day-cadence pattern is a strict subset of
        # the surface pattern (both start with ``augment ``), so try
        # the more-specific one first.
        m = _RE_AUG_DAY.match(w)
        if m is not None:
            cmip = m.group(1)
            attempted.add(CMIP_TO_OUTPUT_RENAMES.get(cmip, cmip))
            continue
        m = _RE_AUG_SURFACE.match(w)
        if m is not None:
            attempted.add(m.group(1))
    return sorted(attempted - present)


def _apply(zarr_path: str, sidecar: dict) -> dict:
    del zarr_path  # sidecar-only migration; no zarr touch
    warnings = sidecar.get("warnings", []) or []
    present = set(sidecar.get("variables_present", []) or [])
    failed = _extract_failed_augment_vars(warnings, present)
    logging.info("  backfilling %d failed-augment variable(s)", len(failed))

    sidecar["esgf_failed_augment_variables"] = failed
    sidecar["schema_version"] = "0.8.1"
    sidecar.setdefault("migrations", []).append(
        {
            "from": "0.8.0",
            "to": "0.8.1",
            "backfilled_failed_augment_variables": failed,
        }
    )
    return sidecar


MIGRATION = Migration(
    from_version="0.8.0",
    to_version="0.8.1",
    description=(
        "Backfill esgf_failed_augment_variables from sidecar warnings "
        "so future augment passes skip variables a prior pass tried "
        "and failed. Sidecar-only."
    ),
    apply=_apply,
)
