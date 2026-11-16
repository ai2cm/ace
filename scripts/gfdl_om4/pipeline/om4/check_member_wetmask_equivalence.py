"""Check that every ensemble member derives the same ocean wetmask.

Each member's wetmask comes from its own snapshot store's reference variable
(the NaN pattern of e.g. ``thetao`` at the first timestep). Members of one
ensemble run on the same grid and bathymetry, and downstream training and
analysis assume their output stores share one mask — so a difference is a
stop-and-report finding about the sources, not something to conform around.
"""

import argparse
import logging

from .config import load_config
from .run import LEVEL_DIM, load_wetmask

logger = logging.getLogger(__name__)


def check_member_wetmask_equivalence(config_path: str, members: list[str]) -> None:
    reference_member = members[0]
    reference = None
    for member in members:
        config = load_config(config_path, member)
        logger.info("loading wetmask for %s from %s", member, config.wetmask.store)
        mask = load_wetmask(config)
        if reference is None:
            reference = mask
            continue
        if reference.sizes != mask.sizes:
            raise AssertionError(
                f"wetmask shape of {member} ({dict(mask.sizes)}) differs from "
                f"{reference_member} ({dict(reference.sizes)})"
            )
        differing = int((reference.values != mask.values).sum())
        if differing:
            raise AssertionError(
                f"wetmask of {member} differs from {reference_member} at "
                f"{differing} cells; the sources do not share a mask — stop "
                "and report, do not conform around this"
            )
    assert reference is not None
    logger.info(
        "wetmasks identical across %d members at all %d levels "
        "(%d ocean cells at the surface)",
        len(members),
        reference.sizes[LEVEL_DIM],
        int(reference.isel({LEVEL_DIM: 0}).sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assert every ensemble member derives the same wetmask."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the pipeline YAML config"
    )
    parser.add_argument(
        "--members", nargs="+", required=True, help="Ensemble-member names to compare"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # apache_beam may have already configured the root logger on import,
    # making basicConfig a no-op; raise the level explicitly.
    logging.getLogger().setLevel(logging.INFO)
    if len(args.members) < 2:
        parser.error("at least two members are needed to compare wetmasks")
    check_member_wetmask_equivalence(args.config, args.members)


if __name__ == "__main__":
    main()
