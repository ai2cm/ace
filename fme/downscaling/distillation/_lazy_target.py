# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Small helper for inspecting FastGen ``LazyCall`` configs.

Kept free of FastGen imports so it is testable without FastGen installed.
"""

from typing import Any


def lazy_target_name(target: Any) -> str:
    """Human-readable name of a FastGen ``LazyCall`` ``_target_``.

    ``LazyCall`` stores ``_target_`` as the class/callable object itself (only
    dataclasses are stringified), so calling a string method such as
    ``endswith`` on it raises ``AttributeError``. This normalizes either form —
    a class object or an already-stringified dotted path — to a name string.
    """
    return getattr(target, "__name__", "") or str(target)
