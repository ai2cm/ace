"""Beaker submission settings shared by this directory's submit_*.py scripts.

Six scripts here launch jobs through the run-ace-*.sh wrappers, and each one
spelled the same workspace/cluster/priority defaults into its own argparse
block and then copied them into the same three environment variables. Those are
one team policy rather than six independent choices, so they live here: moving
the fleet to a different cluster is one edit, and it cannot leave one family
pointed somewhere else.

Defaults target the balancer's managed workspace. ``ai2/ace`` is the workspace
``scripts/beaker_balancer/balance.py`` modifies -- in ``ai2/climate-titan`` it
only counts urgent slots and never touches a job. ``ai2/titan`` is the B200
cluster this directory's runs are sized for; it holds the smaller half of the
allocation (32 slots against jupiter's 72), so urgent is scarcer there.
"""

import argparse
import os

DEFAULT_WORKSPACE = "ai2/ace"
DEFAULT_CLUSTERS = ["ai2/titan"]
DEFAULT_PRIORITY = "urgent"

# Priorities the balancer accepts as a CM_PRIORITY label, plus "none" to submit
# without one. Opting out has to be expressible: an unset variable is what tells
# the balancer a job is not its to move, so there would otherwise be no way to
# submit a job it leaves alone. "immediate" is deliberately absent -- Beaker
# requires a human-supplied reason for it, and it is not a level the balancer
# hands out.
CM_PRIORITY_CHOICES = ["low", "normal", "high", "urgent", "none"]
DEFAULT_CM_PRIORITY = "urgent"
NO_CM_PRIORITY = "none"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared beaker submission options to ``parser``."""
    parser.add_argument(
        "--beaker-workspace",
        default=DEFAULT_WORKSPACE,
        help=f"Beaker workspace to submit jobs to (default: {DEFAULT_WORKSPACE}).",
    )
    parser.add_argument(
        "--beaker-cluster",
        nargs="+",
        default=list(DEFAULT_CLUSTERS),
        metavar="CLUSTER",
        help=(
            "Beaker cluster(s) to target (ex: ai2/titan ai2/jupiter; default: "
            f"{' '.join(DEFAULT_CLUSTERS)})."
        ),
    )
    parser.add_argument(
        "--beaker-priority",
        default=DEFAULT_PRIORITY,
        help=f"Beaker job priority (ex: high or urgent; default: {DEFAULT_PRIORITY}).",
    )
    parser.add_argument(
        "--cm-priority",
        choices=CM_PRIORITY_CHOICES,
        default=DEFAULT_CM_PRIORITY,
        help=(
            "CM_PRIORITY label the balancer manages the job by: its rank when "
            "urgent slots are scarce, and the priority it is dropped to when it "
            f"does not get one (default: {DEFAULT_CM_PRIORITY}). "
            f"'{NO_CM_PRIORITY}' submits without the label, leaving the job "
            "unmanaged. Note a job labelled urgent rests at high, and that the "
            "label is a ranking rather than a ceiling — see "
            "scripts/beaker_balancer/README.md."
        ),
    )


def env(args: argparse.Namespace, **extra: str) -> dict[str, str]:
    """Environment for a run-ace-*.sh invocation: os.environ + the settings.

    ``extra`` carries the per-family variables the wrappers also read (
    ``WANDB_PROJECT``, ``N_GPUS``, ``BEAKER_SHARED_MEMORY``, ``SKIP_VALIDATE``)
    and is applied last, so a caller can override any of these.
    """
    settings = {
        "BEAKER_WORKSPACE": args.beaker_workspace,
        "BEAKER_CLUSTER": " ".join(args.beaker_cluster),
        "BEAKER_PRIORITY": args.beaker_priority,
        # Empty rather than absent: the wrappers default CM_PRIORITY to urgent
        # when it is unset, so opting out has to say so explicitly. They pass
        # no --env for an empty value.
        "CM_PRIORITY": ("" if args.cm_priority == NO_CM_PRIORITY else args.cm_priority),
    }
    return {**os.environ, **settings, **extra}
