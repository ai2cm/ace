"""Shared helpers for parallel tests."""

import os

import pytest

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

requires_parallel = pytest.mark.skipif(
    WORLD_SIZE < 2, reason="requires at least 2 parallel workers (torchrun)"
)


def requires_world_size(n):
    """Skip test unless running with exactly n parallel workers."""
    return pytest.mark.skipif(
        WORLD_SIZE != n, reason=f"requires exactly {n} parallel workers"
    )
