"""Enforce that ``torch_harmonics`` is only imported via ``fme.core.sht``.

Every ``fme/`` module outside the allowlist below must use the public
``fme.core.sht`` API rather than importing ``torch_harmonics`` directly.
This keeps the upstream dependency behind a single facade so that forks,
patches, and distributed wrappers stay centralized.
"""

import ast
import os
from pathlib import Path

# Directories and files that may legitimately import torch_harmonics.
_ALLOWED = {
    # The sht package itself wraps torch_harmonics.
    "fme/core/sht/",
    # The disco fork references upstream for comparison.
    "fme/core/disco/",
    # Patches distributed communication primitives used by torch_harmonics.
    "fme/core/distributed/_gloo_patch.py",
    # Needs the upstream class to verify fme's fork matches it.
    "fme/ace/models/makani_fcn3/models/networks/test_fourcastnet3.py",
}

_FME_ROOT = Path(__file__).resolve().parents[2]  # fme/


def _is_allowed(rel_path: str) -> bool:
    for entry in _ALLOWED:
        if entry.endswith("/"):
            if rel_path.startswith(entry):
                return True
        elif rel_path == entry:
            return True
    return False


def _collect_violators() -> list[str]:
    violators = []
    for dirpath, _dirnames, filenames in os.walk(_FME_ROOT):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, _FME_ROOT.parent)  # relative to fme/..
            if _is_allowed(rel):
                continue
            try:
                source = Path(full).read_text()
                tree = ast.parse(source, filename=full)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "torch_harmonics" or alias.name.startswith(
                            "torch_harmonics."
                        ):
                            violators.append(f"{rel}:{node.lineno}")
                            break
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (
                        node.module == "torch_harmonics"
                        or node.module.startswith("torch_harmonics.")
                    ):
                        violators.append(f"{rel}:{node.lineno}")
    return sorted(violators)


def test_no_direct_torch_harmonics_imports():
    """Files outside the allowlist must not import torch_harmonics directly."""
    violators = _collect_violators()
    assert violators == [], (
        "The following files import torch_harmonics directly instead of "
        "using fme.core.sht:\n  " + "\n  ".join(violators)
    )
