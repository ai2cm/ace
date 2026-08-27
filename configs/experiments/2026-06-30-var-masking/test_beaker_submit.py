"""Tests for the shared beaker submission settings.

These lock the property the module exists for: every submit script in this
directory offers the same options and puts the same values in the environment,
so the cluster the fleet targets and the balancer label it carries cannot drift
between families.
"""

import argparse
import importlib
import inspect
import os

import beaker_submit
import pytest

# Every script that launches jobs through a run-ace-*.sh wrapper.
SUBMIT_MODULES = [
    "submit_cooldown_jobs",
    "submit_eval_jobs",
    "submit_finetune_jobs",
    "submit_mask_jobs",
    "submit_seed_jobs",
    "submit_sst_jobs",
]


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    beaker_submit.add_arguments(parser)
    return parser.parse_args(argv)


def test_defaults_target_the_balancer_managed_workspace():
    args = _parse([])
    assert args.beaker_workspace == "ai2/ace"
    assert args.beaker_cluster == ["ai2/titan"]
    assert args.beaker_priority == "urgent"
    assert args.cm_priority == "urgent"


def test_env_carries_the_settings_the_wrappers_read():
    env = beaker_submit.env(_parse([]))
    assert env["BEAKER_WORKSPACE"] == "ai2/ace"
    assert env["BEAKER_CLUSTER"] == "ai2/titan"
    assert env["BEAKER_PRIORITY"] == "urgent"
    assert env["CM_PRIORITY"] == "urgent"
    # os.environ is inherited, since the wrappers need PATH and BEAKER_TOKEN.
    assert env["PATH"] == os.environ["PATH"]


def test_several_clusters_become_one_space_separated_value():
    env = beaker_submit.env(_parse(["--beaker-cluster", "ai2/titan", "ai2/jupiter"]))
    assert env["BEAKER_CLUSTER"] == "ai2/titan ai2/jupiter"


def test_opting_out_sets_an_empty_label_rather_than_dropping_it():
    """An unset CM_PRIORITY would take the wrapper's default of urgent.

    The wrappers use ``${CM_PRIORITY-urgent}``, so only an explicitly empty
    value submits a job the balancer leaves alone.
    """
    env = beaker_submit.env(_parse(["--cm-priority", "none"]))
    assert env["CM_PRIORITY"] == ""


def test_immediate_is_not_a_label_the_scripts_can_hand_out():
    with pytest.raises(SystemExit):
        _parse(["--cm-priority", "immediate"])


def test_extra_values_are_applied_last():
    env = beaker_submit.env(
        _parse([]), WANDB_PROJECT="VarMasking8", BEAKER_PRIORITY="high"
    )
    assert env["WANDB_PROJECT"] == "VarMasking8"
    assert env["BEAKER_PRIORITY"] == "high"


@pytest.mark.parametrize("module_name", SUBMIT_MODULES)
def test_every_submit_script_uses_the_shared_settings(module_name):
    """A new submit script must route through here rather than reimplement it."""
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    assert "beaker_submit.add_arguments(parser)" in source
    assert "beaker_submit.env(" in source
    # Reimplementing any of the three would silently shadow the shared default.
    assert '"--beaker-workspace"' not in source
    assert '"--beaker-priority"' not in source
    assert '"BEAKER_CLUSTER":' not in source
