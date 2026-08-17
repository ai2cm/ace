import shlex
import subprocess

import click
import dacite
import pytest
import yaml
from click.testing import CliRunner
from copy_to_weka import (
    _bash_command,
    _copy_pairs,
    _gantry_arguments,
    _sources_from_config,
    main,
)
from get_stats import Config

RAW_DIRECTORY = "gs://bucket/native"
COARSENED_DIRECTORY = "gs://bucket/daily"
DESTINATION = "/climate-default/target"


def _config_data(
    include_time_coarsen: bool = True, trailing_slash: bool = False
) -> dict:
    config_data: dict = {
        "runs": {"run-a": "", "run-b": ""},
        "data_output_directory": RAW_DIRECTORY + ("/" if trailing_slash else ""),
        "stats": {
            "output_directory": "gs://bucket/native-stats",
            "data_type": "ERA5",
        },
    }
    if include_time_coarsen:
        config_data["time_coarsen"] = {
            "data_output_directory": COARSENED_DIRECTORY,
            "stats_output_directory": "gs://bucket/daily-stats",
            "factor": 4,
        }
    return config_data


def _config(**kwargs) -> Config:
    return dacite.from_dict(data_class=Config, data=_config_data(**kwargs))


def _write_config(tmp_path, **kwargs) -> str:
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(_config_data(**kwargs)))
    return str(path)


def test_sources_from_config_defaults_to_all_runs():
    sources = _sources_from_config(_config(), coarsened=False, runs=())
    assert sources == [
        RAW_DIRECTORY + "/run-a.zarr",
        RAW_DIRECTORY + "/run-b.zarr",
    ]


def test_sources_from_config_tolerates_trailing_slash():
    sources = _sources_from_config(
        _config(trailing_slash=True), coarsened=False, runs=()
    )
    assert sources == [
        RAW_DIRECTORY + "/run-a.zarr",
        RAW_DIRECTORY + "/run-b.zarr",
    ]


def test_sources_from_config_selects_coarsened_stores():
    sources = _sources_from_config(_config(), coarsened=True, runs=("run-b",))
    assert sources == [COARSENED_DIRECTORY + "/run-b.zarr"]


def test_sources_from_config_rejects_unknown_run():
    with pytest.raises(click.ClickException, match="not in the config"):
        _sources_from_config(_config(), coarsened=False, runs=("run-c",))


def test_sources_from_config_rejects_coarsened_without_section():
    with pytest.raises(click.ClickException, match="no time_coarsen section"):
        _sources_from_config(
            _config(include_time_coarsen=False), coarsened=True, runs=()
        )


def test_sources_from_config_rejects_a_config_with_no_runs():
    config = dacite.from_dict(
        data_class=Config,
        data={
            "runs": {},
            "data_output_directory": RAW_DIRECTORY,
            "stats": {
                "output_directory": "gs://bucket/native-stats",
                "data_type": "ERA5",
            },
        },
    )
    with pytest.raises(click.ClickException, match="no runs to copy"):
        _sources_from_config(config, coarsened=False, runs=())


def test_copy_pairs_appends_source_store_name():
    pairs = _copy_pairs([RAW_DIRECTORY + "/run-a.zarr"], DESTINATION)
    assert pairs[0].destination == DESTINATION + "/run-a.zarr"


def test_copy_pairs_rejects_destination_outside_weka_mount():
    with pytest.raises(click.ClickException, match="or a path under it"):
        _copy_pairs([RAW_DIRECTORY + "/run-a.zarr"], "/net/nfs/target")


@pytest.mark.parametrize("destination", ["/climate-default", "/climate-default/"])
def test_copy_pairs_accepts_the_weka_mount_root(destination):
    """Stores are copied straight to /climate-default in the common case."""
    pairs = _copy_pairs([RAW_DIRECTORY + "/run-a.zarr"], destination)
    assert pairs[0].destination == "/climate-default/run-a.zarr"


def test_copy_pairs_rejects_source_without_a_store_name():
    with pytest.raises(click.ClickException, match="determine a store name"):
        _copy_pairs(["gs://"], DESTINATION)


def test_copy_pairs_rejects_non_gcs_source():
    with pytest.raises(click.ClickException, match="must start with gs://"):
        _copy_pairs(["/climate-default/run-a.zarr"], DESTINATION)


def test_bash_command_fails_on_existing_destination():
    pairs = _copy_pairs([RAW_DIRECTORY + "/run-a.zarr"], DESTINATION)
    command = _bash_command(pairs, overwrite=False)
    quoted = shlex.quote(DESTINATION + "/run-a.zarr")
    assert f"if [ -e {quoted} ]; then" in command
    assert "exit 1" in command
    assert "rm -rf" not in command


def test_bash_command_removes_existing_destination_when_overwriting():
    pairs = _copy_pairs([RAW_DIRECTORY + "/run-a.zarr"], DESTINATION)
    command = _bash_command(pairs, overwrite=True)
    quoted = shlex.quote(DESTINATION + "/run-a.zarr")
    assert f"rm -rf {quoted}" in command
    assert "exit 1" not in command


def test_bash_command_quotes_paths_needing_escaping():
    pairs = _copy_pairs(["gs://bucket/odd name.zarr"], DESTINATION)
    command = _bash_command(pairs, overwrite=False)
    assert "'gs://bucket/odd name.zarr'" in command
    assert f"'{DESTINATION}/odd name.zarr'" in command


def test_bash_command_checks_all_destinations_before_copying():
    sources = [RAW_DIRECTORY + "/run-a.zarr", RAW_DIRECTORY + "/run-b.zarr"]
    command = _bash_command(_copy_pairs(sources, DESTINATION), overwrite=False)
    last_check = command.rindex("if [ -e ")
    first_copy = command.index("gsutil")
    assert last_check < first_copy


def test_gantry_arguments_mount_weka_and_pass_command():
    pairs = _copy_pairs([RAW_DIRECTORY + "/run-a.zarr"], DESTINATION)
    arguments = _gantry_arguments(
        pairs=pairs,
        command="echo hello",
        name="copy-to-weka-dataset",
        workspace="ai2/ace",
        cluster="ai2/phobos",
        priority="normal",
        budget="ai2/atec-climate",
    )
    assert arguments[:2] == ["gantry", "run"]
    assert "climate-default:/climate-default" in arguments
    assert arguments[-3:] == ["bash", "-c", "echo hello"]


def _invoke(arguments, monkeypatch):
    """Run the CLI with source existence checks and job submission stubbed out."""
    monkeypatch.setattr("copy_to_weka._missing_sources", lambda pairs: [])
    return CliRunner().invoke(main, arguments)


def test_cli_dry_run_reports_resolved_paths(tmp_path, monkeypatch):
    config_yaml = _write_config(tmp_path)
    result = _invoke(
        ["--config", config_yaml, "--destination", DESTINATION, "--dry-run"],
        monkeypatch,
    )
    assert result.exit_code == 0, result.output
    assert f"{RAW_DIRECTORY}/run-a.zarr -> {DESTINATION}/run-a.zarr" in result.output
    assert "Dry run, not submitting" in result.output


def test_cli_dry_run_uses_coarsened_store_names(tmp_path, monkeypatch):
    config_yaml = _write_config(tmp_path)
    result = _invoke(
        [
            "--config",
            config_yaml,
            "--destination",
            DESTINATION,
            "--coarsened",
            "--run",
            "run-a",
            "--dry-run",
        ],
        monkeypatch,
    )
    assert result.exit_code == 0, result.output
    assert f"{COARSENED_DIRECTORY}/run-a.zarr" in result.output
    assert RAW_DIRECTORY not in result.output


def test_cli_requires_exactly_one_source_of_stores(tmp_path, monkeypatch):
    config_yaml = _write_config(tmp_path)
    both = _invoke(
        [
            "--config",
            config_yaml,
            "--source",
            RAW_DIRECTORY + "/run-a.zarr",
            "--destination",
            DESTINATION,
        ],
        monkeypatch,
    )
    assert both.exit_code != 0
    assert "exactly one of --config or --source" in both.output

    neither = _invoke(["--destination", DESTINATION], monkeypatch)
    assert neither.exit_code != 0
    assert "exactly one of --config or --source" in neither.output


def test_cli_rejects_run_selection_with_explicit_source(monkeypatch):
    result = _invoke(
        [
            "--source",
            RAW_DIRECTORY + "/run-a.zarr",
            "--destination",
            DESTINATION,
            "--run",
            "run-a",
        ],
        monkeypatch,
    )
    assert result.exit_code != 0
    assert "only apply when copying from a --config" in result.output


def test_cli_submits_gantry_job(tmp_path, monkeypatch):
    config_yaml = _write_config(tmp_path)
    submitted = {}

    def fake_run(arguments, **kwargs):
        submitted["arguments"] = arguments
        submitted["kwargs"] = kwargs
        return subprocess.CompletedProcess(arguments, returncode=0)

    monkeypatch.setattr("copy_to_weka._repository_root", lambda: str(tmp_path))
    monkeypatch.setattr("copy_to_weka.subprocess.run", fake_run)
    result = _invoke(
        ["--config", config_yaml, "--destination", DESTINATION], monkeypatch
    )
    assert result.exit_code == 0, result.output
    assert submitted["arguments"][0] == "gantry"
    assert "copy-to-weka-dataset" in submitted["arguments"]
    assert submitted["kwargs"]["cwd"] == str(tmp_path)
    assert submitted["kwargs"]["check"] is True
    command = submitted["arguments"][-1]
    assert f"{DESTINATION}/run-a.zarr" in command
    assert f"{DESTINATION}/run-b.zarr" in command
