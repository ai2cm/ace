"""Shared helpers for selecting FM configs by version tag (-v1 / -v2).

The FM configs embed a version tag (`-v1`, `-v2`) in their filenames, e.g.
`ace-train-config-4deg-AIMIP-nc-sfno-fm-0.1-v1.yaml` and its cooldown variant
`...-nc-sfno-fm-0.1-v1-cooldown.yaml`. The submit/generate scripts take a
`--version`/`-v` argument to restrict processing to a single version's configs.
"""

import argparse

VERSION_CHOICES = ("v1", "v2")
DEFAULT_VERSION = "v1"


def add_version_arg(parser: argparse.ArgumentParser) -> None:
    """Register the `--version`/`-v` argument on an argument parser."""
    parser.add_argument(
        "-v",
        "--version",
        choices=VERSION_CHOICES,
        default=DEFAULT_VERSION,
        help=(
            "Config version to process: 'v1' selects -v1 configs, "
            f"'v2' selects -v2 configs (default: {DEFAULT_VERSION})."
        ),
    )


def stem_matches_version(stem: str, version: str) -> bool:
    """True if a config stem carries the given version tag as a segment.

    Matches '-v1' in both 'nc-sfno-fm-0.1-v1' and 'nc-sfno-fm-0.1-v1-cooldown',
    but not a longer tag such as '-v12'.
    """
    tag = f"-{version}"
    idx = stem.find(tag)
    if idx == -1:
        return False
    rest = stem[idx + len(tag) :]
    return rest == "" or rest.startswith("-")
