import pathlib

import dacite
import yaml

import fme
from fme.coupled.inference.evaluator import (
    InferenceEvaluatorConfig as CoupledInferenceEvaluatorConfig,
)
from fme.coupled.train.train_config import TrainConfig as CoupledTrainConfig
from fme.downscaling.evaluator import EvaluatorConfig
from fme.downscaling.train import TrainerConfig as DownscalingTrainConfig

EXAMPLES_DIRECTORY = pathlib.Path(__file__).parent


def get_yaml_files(pattern, exclude=None):
    """Get all files matching the pattern in the directory and subdirectories."""
    paths = list(EXAMPLES_DIRECTORY.rglob(pattern))
    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        for exc in exclude:
            paths = [p for p in paths if exc not in str(p)]
    paths = [p for p in paths if "experiments/" not in str(p)]
    return paths


def validate_config(file_path, config_class):
    """Validate the YAML file against the given data class."""
    try:
        with open(file_path) as f:
            data = yaml.safe_load(f)
        dacite.from_dict(
            data_class=config_class, data=data, config=dacite.Config(strict=True)
        )
    except Exception as e:
        raise RuntimeError(f"Validation failed for {file_path}: {e}")


def test_train_configs_are_valid():
    train_files = get_yaml_files(
        "*train*.yaml",
        exclude=[
            "baselines/downscaling",
            "cm4-piControl/train-config",
            "cm4-piControl/finetune-config",
        ],
    )
    assert len(train_files) > 0, "No train files found"
    for file in train_files:
        validate_config(file, fme.ace.TrainConfig)


def test_evaluator_configs_are_valid():
    evaluator_files = get_yaml_files(
        "*evaluator*.yaml",
        exclude=["baselines/downscaling", "cm4-piControl/evaluator-config"],
    )
    assert len(evaluator_files) > 0, "No evaluator files found"
    for file in evaluator_files:
        validate_config(file, fme.ace.InferenceEvaluatorConfig)


def test_coupled_train_configs_are_valid():
    train_files = get_yaml_files(
        "**/cm4-piControl/train-config.yaml",
    ) + get_yaml_files(
        "**/cm4-piControl/finetune-config.yaml",
    )
    assert len(train_files) > 0, "No coupled train files found"
    for file in train_files:
        validate_config(file, CoupledTrainConfig)


def test_coupled_evaluator_configs_are_valid():
    evaluator_files = get_yaml_files("**/cm4-piControl/*evaluator*.yaml")
    assert len(evaluator_files) > 0, "No coupled evaluator files found"
    for file in evaluator_files:
        validate_config(file, CoupledInferenceEvaluatorConfig)


def test_downscaling_train_configs_are_valid():
    downscaling_files = get_yaml_files("**/downscaling/*train*.yaml") + get_yaml_files(
        "**/downscaling-hiro-global/*train*.yaml"
    )
    for file in downscaling_files:
        validate_config(file, DownscalingTrainConfig)


def test_downscaling_evaluator_configs_are_valid():
    downscaling_files = get_yaml_files("**/downscaling/*eval*.yaml")
    for file in downscaling_files:
        validate_config(file, EvaluatorConfig)


def test_inference_configs_are_valid():
    inference_files = get_yaml_files("*inference*.yaml")
    assert len(inference_files) > 0, "No inference files found"
    for file in inference_files:
        validate_config(file, fme.ace.InferenceConfig)
