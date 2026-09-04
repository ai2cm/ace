import os

import dacite
import pytest
import yaml
from append_dataset import DatasetAppendConfig
from combine_stats import Config as CombineStatsConfig
from create_coupled_datasets import CreateCoupledDatasetsConfig
from create_coupled_ic import CreateCoupledICConfig
from get_stats import Config as GetStatsConfig
from time_coarsen import Config as TimeCoarsenConfig
from upload_stats import Config as UploadStatsConfig
from upload_stats import _upload_specs

DIRNAME = os.path.abspath(os.path.dirname(__file__))
CONFIGS_DIR = os.path.join(DIRNAME, "configs")

# Configs whose schema has no test here; any other unclassified config fails
# test_every_config_is_classified.
UNTESTED_CONFIG_SUFFIXES = ("-vertical-coarsen.yaml",)


def _config_kind(path: str) -> str:
    """Classify a config by its top-level keys, not its filename."""
    with open(path) as f:
        keys = set(yaml.load(f, Loader=yaml.CLoader))
    if "coupled_datasets" in keys:
        return "coupled"
    if "coupled_config_path" in keys:
        return "coupled-ic"
    if "variable_sources" in keys:
        return "append"
    if "runs" in keys:
        return "stats"
    return "other"


ALL_CONFIG_YAMLS = [
    os.path.join(CONFIGS_DIR, f)
    for f in sorted(os.listdir(CONFIGS_DIR))
    if f.endswith(".yaml")
]
CONFIG_KINDS = {path: _config_kind(path) for path in ALL_CONFIG_YAMLS}


def _configs_of_kind(kind: str) -> list[str]:
    return [path for path, k in CONFIG_KINDS.items() if k == kind]


CONFIG_YAMLS = _configs_of_kind("stats")
APPEND_CONFIG_YAMLS = _configs_of_kind("append")
COUPLED_CONFIG_YAMLS = _configs_of_kind("coupled")
COUPLED_IC_CONFIG_YAMLS = _configs_of_kind("coupled-ic")


def test_every_config_is_classified():
    unclassified = [
        os.path.basename(path)
        for path in _configs_of_kind("other")
        if not path.endswith(UNTESTED_CONFIG_SUFFIXES)
    ]
    assert unclassified == []


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


NATIVE_DATA_DIRECTORY = "gs://bucket/native-6hourly"
NATIVE_STATS_DIRECTORY = "gs://bucket/native-stats"
COARSENED_STATS_DIRECTORY = "gs://bucket/native-daily-stats"
COARSENED_DATA_DIRECTORY = "gs://bucket/native-daily"
COARSEN_CLAUSE = "Time coarsened by a factor of"


def _upload_stats_config(
    stats_beaker_dataset: str | None = "native-stats",
    time_coarsen_beaker_dataset: str | None = None,
    include_time_coarsen: bool = False,
    exclude_runs: list[str] | None = None,
) -> UploadStatsConfig:
    stats: dict = {
        "output_directory": NATIVE_STATS_DIRECTORY,
        "data_type": "CM4",
        "start_date": "0151-01-01T06:00:00",
        "end_date": "0351-01-01T00:00:00",
    }
    if stats_beaker_dataset is not None:
        stats["beaker_dataset"] = stats_beaker_dataset
    if exclude_runs is not None:
        stats["exclude_runs"] = exclude_runs
    config_data: dict = {
        "runs": {"run-a": "", "run-b": ""},
        "data_output_directory": NATIVE_DATA_DIRECTORY,
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
    assert NATIVE_DATA_DIRECTORY in specs[0].description
    assert "run-a, run-b" in specs[0].description
    assert "0151-01-01T06:00:00" in specs[0].description
    assert COARSEN_CLAUSE not in specs[0].description


def test_upload_specs_description_omits_excluded_runs():
    specs = _upload_specs(_upload_stats_config(exclude_runs=["run-b"]))
    assert "run-a" in specs[0].description
    assert "run-b" not in specs[0].description


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
    assert NATIVE_DATA_DIRECTORY not in coarsened.description
    assert f"{COARSEN_CLAUSE} 4" in coarsened.description


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


def test_upload_config_rejects_no_datasets_requested():
    with pytest.raises(ValueError, match="No Beaker dataset to upload"):
        _upload_stats_config(stats_beaker_dataset=None, include_time_coarsen=True)


def test_upload_config_rejects_no_datasets_without_time_coarsen():
    with pytest.raises(ValueError, match="No Beaker dataset to upload"):
        _upload_stats_config(stats_beaker_dataset=None)


OUTPUT_NAMES_CONFIG_YAMLS = [
    f for f in CONFIG_YAMLS if "output_names" in open(f).read()
]


@pytest.mark.parametrize("filename", OUTPUT_NAMES_CONFIG_YAMLS)
def test_output_names_resolve_to_distinct_paths(filename):
    with open(filename) as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    config = dacite.from_dict(data_class=TimeCoarsenConfig, data=config_data)
    data_dir = config.data_output_directory.rstrip("/")
    tc_dir = config.time_coarsen.data_output_directory.rstrip("/")
    for run_name in config.runs:
        output_name = config.time_coarsen.output_names.get(run_name, run_name)
        input_zarr = data_dir + "/" + run_name + ".zarr"
        output_zarr = tc_dir + "/" + output_name + ".zarr"
        assert (
            input_zarr != output_zarr
        ), f"Coarsened output path equals input: {input_zarr}"
        assert (
            output_name != run_name
        ), f"output_names should give run {run_name!r} a distinct name"
