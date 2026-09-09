"""Checks that the committed run configs match the generator and parse into
``TrainConfig``.

Run with ``python -m pytest`` on this directory.
"""

import importlib.util
import pathlib

import dacite
import pytest
import yaml

from fme.ace.train.train_config import TrainConfig

HERE = pathlib.Path(__file__).parent
RUN_DIR = HERE / "run_configs"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "swin_efficiency_generate_configs", HERE / "generate_configs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_configs = _load_generator()


@pytest.mark.parametrize("name", sorted(generate_configs.VARIANTS))
def test_committed_config_matches_generator(name: str):
    with open(RUN_DIR / name) as f:
        committed = yaml.safe_load(f)
    assert committed == generate_configs.generate(name)


@pytest.mark.parametrize("name", sorted(generate_configs.VARIANTS))
def test_config_parses_into_train_config(name: str):
    config = dacite.from_dict(
        data_class=TrainConfig,
        data=generate_configs.generate(name),
        config=dacite.Config(strict=True),
    )
    assert config.optimization.float32_matmul_precision == "high"
    assert all(entry.weight > 0.0 for entry in config.inference)
    assert all(
        entry.epochs.step == generate_configs.INFERENCE_EPOCH_STEP
        for entry in config.inference
    )


def test_swin_fast_config_enables_compile_and_skip_projection():
    name = "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-a1-fast.yaml"
    step_config = generate_configs.generate(name)["stepper"]["step"]["config"]
    assert step_config["compile"] is True
    assert step_config["builder"]["config"]["skip_projection"] is True
