"""Submit the backward-bisection arms written by make_bwd_configs.py.

Job names: ace2-fm-test-bisect-bwd-<arm>, wandb group
ace2-fm-swin-bisect-test-2026-11-05, default GPU count of run-ace-train.sh.

Usage: python submit_bwd_jobs.py --arm B0-ref B1-levels1 ... [--dry-run]
"""

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

from _submit_common import check_configs_at_head, submit_job  # noqa: E402
from make_bwd_configs import ARMS, PREFIX  # noqa: E402
from submit_fm_jobs import (  # noqa: E402
    RUN_SCRIPT,
    WANDB_PROJECT,
    add_default_beaker_args,
)

WANDB_GROUP = "ace2-fm-swin-bisect-test-2026-11-05"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", nargs="+", choices=list(ARMS), required=True)
    add_default_beaker_args(parser)
    args = parser.parse_args()
    paths = [HERE / f"{PREFIX}{arm}.yaml" for arm in args.arm]
    if not args.dry_run:
        check_configs_at_head(paths)
    for arm, path in zip(args.arm, paths):
        submit_job(
            RUN_SCRIPT,
            [
                str(path.relative_to(EXP_DIR)),
                f"ace2-fm-test-bisect-bwd-{arm}",
                WANDB_GROUP,
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=EXP_DIR,
        )


if __name__ == "__main__":
    main()
