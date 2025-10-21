import os

import dacite
import pytest
import yaml
from append_dataset import DatasetAppendConfig
from combine_stats import Config as CombineStatsConfig
from get_stats import Config as GetStatsConfig
from upload_stats import Config as UploadStatsConfig

DIRNAME = os.path.abspath(os.path.dirname(__file__))
# list files in DIRNAME/config
APPEND_CONFIG_YAMLS = [
    os.path.join(DIRNAME + "/configs", f)
    for f in os.listdir(DIRNAME + "/configs")
    if f.endswith(".yaml") and "append" in f
]
IGNORE_CONFIG_YAMLS = [
    "CM4-piControl-sea-ice-1deg-200yr.yaml",
    "CM4-piControl-200yr-sic-6h-to-5Davg.yaml",
    "CM4-piControl-200yr-ts-6h-to-5Davg.yaml",
    "CM4-piControl-200yr-5daily-sfc-flux.yaml",
    "E3SMv3-piControl-100yr-5daily-sfc-flux.yaml",
    "E3SMv3-piControl-100yr-sic-6h-to5Davg.yaml",
    "e3smv3-piControl-100yr-ts-6h-to-5Davg.yaml",
]
CONFIG_YAMLS = [
    os.path.join(DIRNAME + "/configs", f)
    for f in os.listdir(DIRNAME + "/configs")
    if f.endswith(".yaml") and "append" not in f and f not in IGNORE_CONFIG_YAMLS
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
