import dacite
import pytest
from get_stats import Config, stats_path, store_path


def _config(trailing_slash: bool = False, include_time_coarsen: bool = True) -> Config:
    slash = "/" if trailing_slash else ""
    config_data: dict = {
        "runs": {"run-a": "", "run-b": ""},
        "data_output_directory": "gs://bucket/native" + slash,
        "stats": {
            "output_directory": "gs://bucket/native-stats" + slash,
            "data_type": "ERA5",
        },
    }
    if include_time_coarsen:
        config_data["time_coarsen"] = {
            "data_output_directory": "gs://bucket/daily" + slash,
            "stats_output_directory": "gs://bucket/daily-stats" + slash,
            "factor": 4,
        }
    return dacite.from_dict(data_class=Config, data=config_data)


def test_store_path_appends_zarr_suffix():
    assert store_path("gs://bucket/native", "run-a") == "gs://bucket/native/run-a.zarr"


def test_stats_path_does_not_append_zarr_suffix():
    assert stats_path("gs://bucket/stats", "run-a") == "gs://bucket/stats/run-a"


@pytest.mark.parametrize("trailing_slash", [False, True])
def test_paths_are_unaffected_by_a_trailing_slash(trailing_slash):
    config = _config(trailing_slash=trailing_slash)
    assert config.raw_store("run-a") == "gs://bucket/native/run-a.zarr"
    assert config.raw_stats_directory("run-a") == "gs://bucket/native-stats/run-a"
    assert config.coarsened_store("run-a") == "gs://bucket/daily/run-a.zarr"
    assert config.coarsened_stats_directory("run-a") == "gs://bucket/daily-stats/run-a"


def test_run_names_preserve_config_order():
    assert _config().run_names() == ["run-a", "run-b"]


def test_coarsened_paths_require_a_time_coarsen_section():
    config = _config(include_time_coarsen=False)
    with pytest.raises(ValueError, match="No time_coarsen section"):
        config.coarsened_store("run-a")
    with pytest.raises(ValueError, match="No time_coarsen section"):
        config.coarsened_stats_directory("run-a")
