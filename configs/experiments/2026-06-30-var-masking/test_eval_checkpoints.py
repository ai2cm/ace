"""Tests for the eval checkpoint registry.

These cover the invariants the registry exists to hold: that a checkpoint added
to it is visible to every consumer, and that the resolved path for an
epoch-numbered checkpoint is the last epoch actually written.
"""

import eval_checkpoints
import pytest
from eval_checkpoints import EVAL_CHECKPOINTS, by_names, is_derived_run_name, names

RUN_NAME = "ace2-var-mask-nc-sfno-era5-gmron-mask20-seed0-v5-mstepft"


def _lister(paths):
    """A DatasetLister returning ``paths``, ignoring the dataset and prefix."""
    return lambda dataset_id, prefix: paths


def test_every_checkpoint_is_selectable_and_counted():
    for checkpoint in EVAL_CHECKPOINTS:
        assert checkpoint.name in names()
        assert checkpoint.suffix in eval_checkpoints.suffixes()
        assert is_derived_run_name(f"{RUN_NAME}{checkpoint.suffix}")


def test_training_run_name_is_not_derived():
    assert not is_derived_run_name(RUN_NAME)


def test_longer_suffix_is_not_read_as_the_shorter_one():
    # -lastepoch is a prefix of -lastepochema, so a run of the latter must be
    # recognised as its own checkpoint rather than as a -lastepoch run.
    lastepochema = by_names(["lastepochema"])[0]
    assert lastepochema.suffix.startswith("-lastepoch")
    assert is_derived_run_name(f"{RUN_NAME}-lastepochema")
    assert f"{RUN_NAME}-lastepochema" not in {
        f"{RUN_NAME}{c.suffix}" for c in EVAL_CHECKPOINTS if c.name != "lastepochema"
    }


def test_by_names_keeps_canonical_order_and_deduplicates():
    selected = by_names(["lastepoch", "besttrain", "lastepoch"])
    assert [checkpoint.name for checkpoint in selected] == ["besttrain", "lastepoch"]


def test_by_names_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown checkpoint"):
        by_names(["bestinf", "nosuchcheckpoint"])


def test_fixed_path_checkpoints_ignore_the_dataset():
    lastepoch = by_names(["lastepoch"])[0]
    assert lastepoch.resolve("some-dataset-id") == "training_checkpoints/ckpt.tar"


def test_latest_ema_checkpoint_takes_the_highest_epoch():
    paths = [
        "training_checkpoints/ema_ckpt_0010.tar",
        "training_checkpoints/ema_ckpt_0150.tar",
        "training_checkpoints/ema_ckpt_0020.tar",
    ]
    assert (
        eval_checkpoints.latest_ema_checkpoint("dataset", _lister(paths))
        == "training_checkpoints/ema_ckpt_0150.tar"
    )


def test_latest_ema_checkpoint_is_none_without_one():
    paths = ["training_checkpoints/ckpt.tar"]
    assert eval_checkpoints.latest_ema_checkpoint("dataset", _lister(paths)) is None
