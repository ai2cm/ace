import pathlib
import typing

import fme.ace
from fme.ace.aggregator.inference.main import MetricConfig

DOCS_DIR = pathlib.Path(__file__).parent


def test_all_metric_configs_documented():
    """Every type in the MetricConfig union must appear in evaluator_config.rst."""
    docs_content = (DOCS_DIR / "evaluator_config.rst").read_text()

    for cls in typing.get_args(MetricConfig):
        name = cls.__name__
        assert hasattr(
            fme.ace, name
        ), f"{name} is in MetricConfig union but not exported from fme.ace"
        assert f"fme.ace.{name}" in docs_content, (
            f"{name} is in MetricConfig union but not documented in "
            f"docs/evaluator_config.rst"
        )
