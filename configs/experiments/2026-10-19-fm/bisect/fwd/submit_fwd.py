"""Submit the forward-bisection training jobs (make_configs.py configs).

Each config ace-train-config-4deg-nc-swin-bisect-fwd-<arm>.yaml runs via
run-ace-train.sh on 4 GPUs as job ace2-fm-test-bisect-fwd-<arm> in wandb
group ace2-fm-swin-bisect-test-2026-11-05.

Usage: python submit_fwd.py [--arm F0-v21 ...] [--dry-run] [--beaker-* ...]
"""

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent.parent
sys.path.insert(0, str(EXP))

from _submit_common import (  # noqa: E402
    check_configs_at_head,
    drop_jobs_in_beaker,
    submit_job,
)
from submit_fm_jobs import (  # noqa: E402
    RUN_SCRIPT,
    WANDB_PROJECT,
    add_default_beaker_args,
)

WANDB_GROUP = "ace2-fm-swin-bisect-test-2026-11-05"
N_GPUS = "4"
PREFIX = "ace-train-config-4deg-nc-swin-bisect-fwd-"


def arm_of(path: pathlib.Path) -> str:
    return path.stem.removeprefix(PREFIX)


def job_name(path: pathlib.Path) -> str:
    return f"ace2-fm-test-bisect-fwd-{arm_of(path)}"


def main() -> None:
    configs = sorted(HERE.glob(f"{PREFIX}*.yaml"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", nargs="+", choices=[arm_of(p) for p in configs], default=None
    )
    add_default_beaker_args(parser)
    args = parser.parse_args()
    if args.arm is not None:
        configs = [p for p in configs if arm_of(p) in args.arm]
    configs = drop_jobs_in_beaker(configs, job_name, args)
    if not args.dry_run:
        check_configs_at_head(configs)
    for path in configs:
        submit_job(
            RUN_SCRIPT,
            [str(path.relative_to(EXP)), job_name(path), WANDB_GROUP],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=EXP,
            extra_env={"N_GPUS": N_GPUS},
        )


if __name__ == "__main__":
    main()
