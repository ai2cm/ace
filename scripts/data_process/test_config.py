import os

import dacite
import pytest
import yaml
from append_dataset import DatasetAppendConfig
from combine_stats import Config as CombineStatsConfig
from create_coupled_datasets import CreateCoupledDatasetsConfig
from create_coupled_ic import CreateCoupledICConfig
from get_stats import Config as GetStatsConfig
from upload_stats import Config as UploadStatsConfig
from upload_stats import _upload_specs

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
    "-coupled-ic.yaml",
]
COUPLED_IC_CONFIG_YAMLS = [
    os.path.join(DIRNAME + "/configs", f)
    for f in os.listdir(DIRNAME + "/configs")
    if f.endswith("-coupled-ic.yaml")
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


@pytest.mark.parametrize("filename", COUPLED_IC_CONFIG_YAMLS)
def test_valid_create_coupled_ic_config(filename):
    with open(filename, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    dacite.from_dict(
        data_class=CreateCoupledICConfig,
        data=config_data,
        config=dacite.Config(cast=[tuple], strict=True),
    )


NATIVE_STATS_DIRECTORY = "gs://bucket/native-stats"
COARSENED_STATS_DIRECTORY = "gs://bucket/native-daily-stats"
COARSENED_DATA_DIRECTORY = "gs://bucket/native-daily"


def _upload_stats_config(
    stats_beaker_dataset: str | None = "native-stats",
    time_coarsen_beaker_dataset: str | None = None,
    include_time_coarsen: bool = False,
) -> UploadStatsConfig:
    stats = {
        "output_directory": NATIVE_STATS_DIRECTORY,
        "data_type": "CM4",
        "start_date": "0151-01-01T06:00:00",
        "end_date": "0351-01-01T00:00:00",
    }
    if stats_beaker_dataset is not None:
        stats["beaker_dataset"] = stats_beaker_dataset
    config_data: dict = {
        "runs": {"run-a": "", "run-b": ""},
        "data_output_directory": "gs://bucket",
        "stats": stats,
    }
    if include_time_coarsen:
        time_coarsen = {
            "data_output_directory": COARSENED_DATA_DIRECTORY,
            "stats_output_directory": COARSENED_STATS_DIRECTORY,
            "factor": 4,
        }
        if time_coarsen_beaker_dataset is not None:
            time_coarsen["beaker_dataset"] = time_coarsen_beaker_dataset
        config_data["time_coarsen"] = time_coarsen
    return dacite.from_dict(data_class=UploadStatsConfig, data=config_data)


def test_upload_specs_native_only():
    specs = _upload_specs(_upload_stats_config())
    assert len(specs) == 1
    assert specs[0].beaker_dataset == "native-stats"
    assert specs[0].combined_directory == NATIVE_STATS_DIRECTORY + "/combined/"
    assert "gs://bucket" in specs[0].description
    assert "run-a, run-b" in specs[0].description
    assert "0151-01-01T06:00:00" in specs[0].description


def test_upload_specs_skips_time_coarsened_without_beaker_dataset():
    specs = _upload_specs(_upload_stats_config(include_time_coarsen=True))
    assert [spec.beaker_dataset for spec in specs] == ["native-stats"]


def test_upload_specs_includes_time_coarsened_dataset():
    specs = _upload_specs(
        _upload_stats_config(
            include_time_coarsen=True,
            time_coarsen_beaker_dataset="native-daily-stats",
        )
    )
    assert [spec.beaker_dataset for spec in specs] == [
        "native-stats",
        "native-daily-stats",
    ]
    coarsened = specs[1]
    assert coarsened.combined_directory == COARSENED_STATS_DIRECTORY + "/combined/"
    assert COARSENED_DATA_DIRECTORY in coarsened.description
    assert "factor of 4" in coarsened.description


def test_upload_specs_time_coarsened_only():
    specs = _upload_specs(
        _upload_stats_config(
            stats_beaker_dataset=None,
            include_time_coarsen=True,
            time_coarsen_beaker_dataset="native-daily-stats",
        )
    )
    assert [spec.beaker_dataset for spec in specs] == ["native-daily-stats"]
    assert specs[0].combined_directory == COARSENED_STATS_DIRECTORY + "/combined/"


def test_upload_specs_no_datasets_requested():
    specs = _upload_specs(
        _upload_stats_config(stats_beaker_dataset=None, include_time_coarsen=True)
    )
    assert specs == []
