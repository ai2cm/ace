"""Checks that the committed run configs match the generator and parse into
``TrainConfig``.

Run with ``python -m pytest`` on this directory.
"""

import importlib.util
import pathlib

import dacite
import pytest
import torch
import yaml

from fme.ace.train.train_config import TrainConfig
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.registry import ModuleSelector

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


# Channel counts of the base config: 45 inputs plus the appended global-mean
# input, 50 outputs plus the shared global-mean channel.
N_IN_CHANNELS = 46
N_OUT_CHANNELS = 51
# Rough per-million bounds for the scaled-down variants (see README).
EXPECTED_PARAM_RANGES_M = {
    "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-a1-fast-compute-matched.yaml": (
        30,
        40,
    ),
    "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-a1-fast-param-matched.yaml": (
        10,
        15,
    ),
    "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-a1-fast-mid.yaml": (42, 52),
}


def _dataset_info_4deg() -> DatasetInfo:
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.linspace(-88.0, 88.0, 45), lon=torch.linspace(0.0, 356.0, 90)
        ),
        all_labels={"amip", "ramped", "som", "era5"},
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_PARAM_RANGES_M))
def test_scaled_down_swin_variants_have_expected_parameter_counts(name: str):
    builder = generate_configs.generate(name)["stepper"]["step"]["config"]["builder"]
    module = ModuleSelector(type=builder["type"], config=builder["config"]).build(
        N_IN_CHANNELS, N_OUT_CHANNELS, _dataset_info_4deg()
    )
    n_params_m = sum(p.numel() for p in module.torch_module.parameters()) / 1e6
    low, high = EXPECTED_PARAM_RANGES_M[name]
    assert low <= n_params_m <= high, n_params_m
    assert module.torch_module.conditional_model.skip_proj is not None
