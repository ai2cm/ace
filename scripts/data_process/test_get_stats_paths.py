import os

import dacite
import pytest
import yaml
from get_stats import Config, stats_path, store_path

DIRNAME = os.path.abspath(os.path.dirname(__file__))


def _config_data(
    raw_directory: str = "gs://bucket/native",
    coarsen_directory: str = "gs://bucket/daily",
    coarsen_stats_directory: str = "gs://bucket/daily-stats",
    output_name: str | None = None,
    runs: tuple[str, ...] = ("run-a", "run-b"),
    include_time_coarsen: bool = True,
) -> dict:
    config_data: dict = {
        "runs": {run: "" for run in runs},
        "data_output_directory": raw_directory,
        "stats": {
            "output_directory": "gs://bucket/native-stats",
            "data_type": "ERA5",
        },
    }
    if include_time_coarsen:
        time_coarsen: dict = {
            "data_output_directory": coarsen_directory,
            "stats_output_directory": coarsen_stats_directory,
            "factor": 4,
            "snapshot_names": [],
            "window_names": [],
            "constant_prefixes": [],
        }
        if output_name is not None:
            time_coarsen["output_name"] = output_name
        config_data["time_coarsen"] = time_coarsen
    return config_data


def _config(**kwargs) -> Config:
    return dacite.from_dict(data_class=Config, data=_config_data(**kwargs))


def test_store_path_appends_zarr_suffix():
    assert store_path("gs://bucket/native", "run-a") == "gs://bucket/native/run-a.zarr"


def test_stats_path_does_not_append_zarr_suffix():
    assert stats_path("gs://bucket/stats", "run-a") == "gs://bucket/stats/run-a"


@pytest.mark.parametrize("trailing_slash", ["", "/"])
def test_paths_are_unaffected_by_a_trailing_slash(trailing_slash):
    config = _config(
        raw_directory="gs://bucket/native" + trailing_slash,
        coarsen_directory="gs://bucket/daily" + trailing_slash,
        coarsen_stats_directory="gs://bucket/daily-stats" + trailing_slash,
    )
    assert config.raw_store("run-a") == "gs://bucket/native/run-a.zarr"
    assert config.coarsened_store("run-a") == "gs://bucket/daily/run-a.zarr"
    assert config.coarsened_stats_directory("run-a") == "gs://bucket/daily-stats/run-a"


def test_run_names_preserve_config_order():
    assert _config().run_names() == ["run-a", "run-b"]


def test_included_run_names_omits_excluded_runs():
    config_data = _config_data()
    config_data["stats"]["exclude_runs"] = ["run-a"]
    config = dacite.from_dict(data_class=Config, data=config_data)
    assert config.included_run_names() == ["run-b"]


def test_coarsened_paths_require_a_time_coarsen_section():
    config = _config(include_time_coarsen=False)
    with pytest.raises(ValueError, match="No time_coarsen section"):
        config.coarsened_store("run-a")
    with pytest.raises(ValueError, match="No time_coarsen section"):
        config.coarsened_stats_directory("run-a")


def test_runs_name_the_coarsened_data_without_an_output_name():
    """Ensemble configs name each coarsened store after its run."""
    config = _config()
    assert config.coarsened_store("run-a") == "gs://bucket/daily/run-a.zarr"
    assert config.coarsened_stats_directory("run-a") == "gs://bucket/daily-stats/run-a"


def test_output_name_replaces_the_run_name_in_store_and_stats():
    config = _config(
        raw_directory="gs://bucket",
        coarsen_directory="gs://bucket",
        runs=("source-dataset",),
        output_name="daily-dataset",
    )
    assert config.raw_store("source-dataset") == "gs://bucket/source-dataset.zarr"
    assert config.coarsened_store("source-dataset") == "gs://bucket/daily-dataset.zarr"
    assert (
        config.coarsened_stats_directory("source-dataset")
        == "gs://bucket/daily-stats/daily-dataset"
    )


@pytest.mark.parametrize("output_name", ["nested/name", "name.zarr"])
def test_output_name_must_be_a_bare_dataset_name(output_name):
    with pytest.raises(ValueError, match="must be a bare dataset name"):
        _config(runs=("run-a",), output_name=output_name)


def test_coarsening_into_a_subdirectory_of_the_input_requires_an_output_name():
    """The #1399 shape: the coarsened store would be named after its input."""
    with pytest.raises(ValueError, match="issues/1399"):
        _config(
            raw_directory="gs://bucket",
            coarsen_directory="gs://bucket/daily-dataset",
            runs=("source-dataset",),
        )


def test_coarsening_into_a_subdirectory_is_allowed_with_an_output_name():
    config = _config(
        raw_directory="gs://bucket",
        coarsen_directory="gs://bucket/daily-dataset",
        runs=("source-dataset",),
        output_name="daily-dataset",
    )
    assert (
        config.coarsened_store("source-dataset")
        == "gs://bucket/daily-dataset/daily-dataset.zarr"
    )


def test_a_sibling_coarsened_directory_needs_no_output_name():
    """Ensemble configs put the coarsened data in a parallel dataset directory."""
    config = _config(
        raw_directory="gs://bucket/dataset",
        coarsen_directory="gs://bucket/dataset-daily",
    )
    assert config.coarsened_store("run-a") == "gs://bucket/dataset-daily/run-a.zarr"


def test_sharing_a_stats_directory_with_the_input_is_rejected():
    """get_stats.py would skip the coarsened stats as already computed."""
    with pytest.raises(ValueError, match="same directory as the stats"):
        _config(coarsen_stats_directory="gs://bucket/native-stats")


def test_sharing_a_stats_directory_is_allowed_when_output_name_separates_them():
    config = _config(
        raw_directory="gs://bucket",
        coarsen_directory="gs://bucket",
        coarsen_stats_directory="gs://bucket/native-stats",
        runs=("source-dataset",),
        output_name="daily-dataset",
    )
    assert config.raw_stats_directory("source-dataset") == (
        "gs://bucket/native-stats/source-dataset"
    )
    assert config.coarsened_stats_directory("source-dataset") == (
        "gs://bucket/native-stats/daily-dataset"
    )


def test_coarsening_over_the_input_is_rejected():
    with pytest.raises(ValueError, match="would overwrite the data it reads"):
        _config(raw_directory="gs://bucket", coarsen_directory="gs://bucket")


def test_an_output_name_matching_a_run_name_is_rejected():
    with pytest.raises(ValueError, match="would overwrite the data it reads"):
        _config(
            raw_directory="gs://bucket",
            coarsen_directory="gs://bucket",
            runs=("run-a",),
            output_name="run-a",
        )


@pytest.mark.parametrize(
    "filename",
    [
        "era5-1deg-8layer-1940-2025.yaml",
        "era5-2deg-8layer-1940-2025.yaml",
        "era5-4deg-8layer-1940-2025.yaml",
        "ufs-replay-ocean-1deg-19level-daily.yaml",
    ],
)
def test_coarsened_stores_are_not_named_after_their_input(filename):
    """Regression test for #1399, asserted without pinning dataset dates.

    Each of these configs has one run whose name is a dataset name, which is the
    case that used to put the input's name on the coarsened store.
    """
    path = os.path.join(DIRNAME, "configs", filename)
    config = dacite.from_dict(data_class=Config, data=yaml.safe_load(open(path)))
    (run,) = config.run_names()
    raw, coarsened = config.raw_store(run), config.coarsened_store(run)
    assert coarsened != raw
    assert run not in coarsened, f"{coarsened} is named after its input {run}"
    # The coarsened store sits beside its input rather than under a directory of
    # its own, so both follow one convention.
    assert os.path.dirname(coarsened) == os.path.dirname(raw)
    assert run not in config.coarsened_stats_directory(run)
    assert config.coarsened_stats_directory(run) != config.raw_stats_directory(run)
