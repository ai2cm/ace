import os

import dacite
import pytest
import yaml
from combine_stats import Config as CombineStatsConfig
from get_stats import Config as GetStatsConfig
from upload_stats import Config as UploadStatsConfig

DIRNAME = os.path.abspath(os.path.dirname(__file__))
# list files in DIRNAME/config
CONFIG_YAMLS = [
    os.path.join(DIRNAME + "/configs", f)
    for f in os.listdir(DIRNAME + "/configs")
    if f.endswith(".yaml")
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
