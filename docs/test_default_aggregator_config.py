import dataclasses
import pathlib

import yaml

from fme.ace.aggregator.inference.main import InferenceEvaluatorAggregatorConfig
from fme.core.testing.regression import validate_text

DOCS_DIR = pathlib.Path(__file__).parent


def test_default_aggregator_config_yaml():
    """Regression test ensuring the default aggregator config YAML stays in sync."""
    config = InferenceEvaluatorAggregatorConfig()
    content = yaml.dump(
        {"aggregator": dataclasses.asdict(config)},
        default_flow_style=False,
        sort_keys=False,
    )
    validate_text(content, DOCS_DIR / "default-aggregator-config.yaml")
