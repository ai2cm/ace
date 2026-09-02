"""Shared helpers for selecting FM configs by version tag (-v1 / -v2 / -v3).

The FM configs embed a version tag (`-v1`, `-v2`, `-v3`) in their filenames,
e.g. `ace-train-config-4deg-AIMIP-nc-sfno-fm-0.1-v1.yaml` and its cooldown
variant `...-nc-sfno-fm-0.1-v1-cooldown.yaml`. Configs named by regime and arm
rather than by version, such as `...-nc-swin-v2-c96-a1.yaml`, carry no version
tag at all and are only selected when no version is requested.

The submit/generate scripts take an optional `--version`/`-v` argument to
restrict processing to a single version's configs; when omitted, all versions
are processed.
"""

import argparse

VERSION_CHOICES = ("v1", "v2", "v3")

# Architecture tags which themselves end in a version-like segment. They are
# removed from a stem before the version tag is looked for, so that the `-v2`
# of `nc-swin-v2` is not read as the config version `v2`.
ARCH_TAGS = ("nc-swin-v2",)


def add_version_arg(parser: argparse.ArgumentParser) -> None:
    """Register the optional `--version`/`-v` argument on an argument parser."""
    parser.add_argument(
        "-v",
        "--version",
        choices=VERSION_CHOICES,
        default=None,
        help=(
            "Config version to process: 'v1' selects -v1 configs, 'v2' selects "
            "-v2 configs, and so on. If omitted, all versions are processed."
        ),
    )


def stem_matches_version(stem: str, version: str | None) -> bool:
    """True if a config stem carries the given version tag as a segment.

    Matches '-v1' in both 'nc-sfno-fm-0.1-v1' and 'nc-sfno-fm-0.1-v1-cooldown',
    but not a longer tag such as '-v12'. A `version` of None matches all stems.

    An ARCH_TAGS occurrence is removed first, so
    'ace-train-config-4deg-AIMIP-nc-swin-v2-c96-a1' carries no version tag and
    matches only `version=None`, while '...-nc-swin-v2-fm-random-v1' still
    matches 'v1'.
    """
    if version is None:
        return True
    for arch_tag in ARCH_TAGS:
        stem = stem.replace(arch_tag, "")
    tag = f"-{version}"
    idx = stem.find(tag)
    if idx == -1:
        return False
    rest = stem[idx + len(tag) :]
    return rest == "" or rest.startswith("-")
