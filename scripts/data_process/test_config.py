import os

import dacite
import pytest
import yaml
from append_dataset import DatasetAppendConfig
from combine_stats import Config as CombineStatsConfig
from create_coupled_datasets import CreateCoupledDatasetsConfig
from get_stats import Config as GetStatsConfig
from upload_stats import Config as UploadStatsConfig

DIRNAME = os.path.abspath(os.path.dirname(__file__))
# list files in DIRNAME/config
APPEND_CONFIG_YAMLS = [
    os.path.join(DIRNAME + "/configs", f)
    for f in os.listdir(DIRNAME + "/configs")
    if f.endswith(".yaml") and "append" in f
]
COUPLED_CONFIG_YAMLS = [
    os.path.join(DIRNAME + "/configs", f)
    for f in os.listdir(DIRNAME + "/configs")
    if f.endswith("-coupled.yaml")
]
IGNORE_CONFIGS_WITH_SUFFIX = [
    "-append.yaml",
    "-coupled.yaml",
    "-vertical-coarsen.yaml",
]


def _ignore_config(fname: str) -> bool:
    return any([fname.endswith(suffix) for suffix in IGNORE_CONFIGS_WITH_SUFFIX])


CONFIG_YAMLS = [
    os.path.join(DIRNAME + "/configs", f)
    for f in os.listdir(DIRNAME + "/configs")
    if f.endswith(".yaml") and not _ignore_config(f)
]


@pytest.mark.parametrize(
    "filename",
    CONFIG_YAMLS,
)
@pytest.mark.parametrize("cls", [GetStatsConfig, UploadStatsConfig, CombineStatsConfig])
def test_get_stats_valid(filename, cls):
    with open(filename, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    dacite.from_dict(data_class=cls, data=config_data)


@pytest.mark.parametrize(
    "filename",
    APPEND_CONFIG_YAMLS,
)
@pytest.mark.parametrize("cls", [DatasetAppendConfig])
def test_valid_dataset_append_config(filename, cls):
    with open(filename, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    dacite.from_dict(data_class=cls, data=config_data)


@pytest.mark.parametrize(
    "filename",
    COUPLED_CONFIG_YAMLS,
)
def test_valid_create_coupled_datasets_config(filename):
    with open(filename, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    dacite.from_dict(data_class=CreateCoupledDatasetsConfig, data=config_data)
