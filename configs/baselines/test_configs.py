import pathlib

import dacite
import yaml

import fme

EXAMPLES_DIRECTORY = pathlib.Path(__file__).parent


def get_yaml_files(pattern):
    """Get all files matching the pattern in the directory and subdirectories."""
    return list(EXAMPLES_DIRECTORY.rglob(pattern))


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
    train_files = get_yaml_files("*train*.yaml")
    assert len(train_files) > 0, "No train files found"
    for file in train_files:
        validate_config(file, fme.ace.TrainConfig)


def test_evaluator_configs_are_valid():
    evaluator_files = get_yaml_files("*evaluator*.yaml")
    assert len(evaluator_files) > 0, "No evaluator files found"
    for file in evaluator_files:
        validate_config(file, fme.ace.InferenceEvaluatorConfig)


def test_inference_configs_are_valid():
    inference_files = get_yaml_files("*inference*.yaml")
    assert len(inference_files) > 0, "No inference files found"
    for file in inference_files:
        validate_config(file, fme.ace.InferenceConfig)
