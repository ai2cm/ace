"""Per-version schema migrations for the CMIP6 archive.

Each migration is a ``Migration`` (defined in ``schema_version.py``)
that takes a dataset's zarr path and its current sidecar dict,
applies whatever on-disk change moves the dataset from
``from_version`` to ``to_version``, and returns the updated sidecar
dict. ``migrate.py`` chains the registered migrations to bring any
old dataset up to the current ``SCHEMA_VERSION``.

Add a new migration by:

1. Writing ``_<from>_to_<to>.py`` next to this file with a
   module-level ``MIGRATION = Migration(...)`` instance.
2. Importing it here and appending to ``MIGRATIONS`` in order.
3. Bumping ``SCHEMA_VERSION`` in ``schema_version.py``.

``MIGRATIONS`` is ordered: ``MIGRATIONS[i].to_version`` ==
``MIGRATIONS[i + 1].from_version`` for the chain to compose.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema_version import Migration, version_lt  # noqa: E402

from migrations._0_0_0_to_0_1_0 import MIGRATION as _M_0_0_0_TO_0_1_0  # noqa: E402
from migrations._0_1_0_to_0_2_0 import MIGRATION as _M_0_1_0_TO_0_2_0  # noqa: E402
from migrations._0_2_0_to_0_3_0 import MIGRATION as _M_0_2_0_TO_0_3_0  # noqa: E402
from migrations._0_3_0_to_0_4_0 import MIGRATION as _M_0_3_0_TO_0_4_0  # noqa: E402
from migrations._0_4_0_to_0_5_0 import MIGRATION as _M_0_4_0_TO_0_5_0  # noqa: E402
from migrations._0_5_0_to_0_6_0 import MIGRATION as _M_0_5_0_TO_0_6_0  # noqa: E402

MIGRATIONS: tuple[Migration, ...] = (
    _M_0_0_0_TO_0_1_0,
    _M_0_1_0_TO_0_2_0,
    _M_0_2_0_TO_0_3_0,
    _M_0_3_0_TO_0_4_0,
    _M_0_4_0_TO_0_5_0,
    _M_0_5_0_TO_0_6_0,
)


def chain_for(current_version: str, target_version: str) -> list[Migration]:
    """Return the migrations that move a dataset from
    ``current_version`` to ``target_version``. Raises ``RuntimeError``
    if no contiguous chain exists between them.
    """
    chain: list[Migration] = []
    cur = current_version
    while version_lt(cur, target_version):
        next_step = next((m for m in MIGRATIONS if m.from_version == cur), None)
        if next_step is None:
            raise RuntimeError(
                f"no migration registered from {cur!r}; expected to reach "
                f"{target_version!r} but the chain stopped here. Check "
                f"``migrations/`` and ``MIGRATIONS`` registration."
            )
        chain.append(next_step)
        cur = next_step.to_version
    return chain


__all__ = ["MIGRATIONS", "Migration", "chain_for"]
