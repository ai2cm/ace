"""Check that two configs derive identical ocean wetmasks.

Each config derives its wetmask from its own snapshot store's reference
variable (the NaN pattern of e.g. ``thetao`` at the first timestep).
Simulations run on the same grid and bathymetry must produce identical
masks, and downstream training and analysis assume their output stores
share one — so a difference is a stop-and-report finding about the
sources, not something to conform around.
"""

import argparse
import logging

import numpy as np

from .config import load_config
from .run import LEVEL_DIM, load_wetmask

logger = logging.getLogger(__name__)


def check_wetmask_equivalence(config_path_a: str, config_path_b: str) -> None:
    masks = {}
    for path in (config_path_a, config_path_b):
        config = load_config(path)
        logger.info("loading wetmask for %s from %s", path, config.wetmask.store)
        masks[path] = load_wetmask(config)
    mask_a, mask_b = masks[config_path_a], masks[config_path_b]
    if mask_a.sizes != mask_b.sizes:
        raise AssertionError(
            f"wetmask shapes differ: {dict(mask_a.sizes)} vs {dict(mask_b.sizes)}"
        )
    if not np.array_equal(mask_a[LEVEL_DIM].values, mask_b[LEVEL_DIM].values):
        raise AssertionError("wetmask level coordinates differ")
    differing = int((mask_a.values != mask_b.values).sum())
    if differing:
        per_level = (mask_a.values != mask_b.values).sum(axis=(-2, -1))
        raise AssertionError(
            f"wetmasks differ at {differing} cells "
            f"(per level: {per_level.tolist()}); the sources do not share a "
            "mask — stop and report, do not conform around this"
        )
    logger.info(
        "wetmasks identical at all %d levels (%d ocean cells at the surface)",
        mask_a.sizes[LEVEL_DIM],
        int(mask_a.isel({LEVEL_DIM: 0}).sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assert two configs derive identical ocean wetmasks."
    )
    parser.add_argument(
        "configs", nargs=2, help="Two pipeline YAML config paths to compare"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # apache_beam may have already configured the root logger on import,
    # making basicConfig a no-op; raise the level explicitly.
    logging.getLogger().setLevel(logging.INFO)
    check_wetmask_equivalence(*args.configs)


if __name__ == "__main__":
    main()
