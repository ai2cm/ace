"""Generate var-masking training configs from the nc-sfno era5 baseline config.

Each generated config name (and thus its wandb run name) ends in ``-v1``,
``-v2``, ``-v3`` or ``-v4``, matching the baseline config version it was
sourced from (see ``baseline_configs/versions.md``); ``--version`` selects
which (default v1). Versions are auto-discovered from ``baseline_configs/``,
so adding a new ``ace2-var-mask-nc-sfno-era5-vN.yaml`` there makes ``vN`` a
valid ``--version`` with no code change.
Full factorial sweep written to ``run_configs/`` (emptied first) over three
axes:

  - uniform mask-level (``default`` scheme, ``max_masked_vars``): 0, 5, 10,
    20, 30.
  - co2 bernoulli masking: off, or drop ``global_mean_co2`` w.p. 0.75.
  - global mean removal (``global_mean_removal`` stepper config): baseline
    (as in the base config) or none (key removed).

5 x 2 x 2 = 20 configs for v1.

global_mean_co2 is already an input channel in the v1 baseline config
(in_names + next_step_forcing_names); the co2 axis is meaningless for v2,
v3 and v4, which drop it as an input entirely (see
baseline_configs/versions.md), so those versions only generate the co2default
option: 5 x 2 x 1 = 10 configs.
"""

import argparse
import copy
import pathlib
import re

import yaml

WANDB_ENTITY = "ai2cm"
WANDB_PROJECT = "VarMasking8"
WANDB_PREFIX = "ace2-var-mask-"  # stripped from wandb run names before comparison
CONFIG_PREFIX = "ace-train-config-4deg-"  # stripped from config stems

# Generated config filename prefix (independent of the source baseline filename
# below).
BASE_CONFIG_STEM = "ace-train-config-4deg-nc-sfno-era5"

CO2_FIELD = "global_mean_co2"
GMR_FIELD = "global_mean_removal"
SPECTRAL_BAND_FIELDS = ("filter_num_groups", "spectral_ratio")

MASK_LEVELS = [0, 5, 10, 20, 30]  # uniform default max_masked_vars
CO2_OPTIONS = {"co2default": None, "co2bern75": 0.75}
GMR_OPTIONS = {"gmron": True, "gmroff": False}  # True: keep baseline config


def co2_options_for_version(version: str) -> dict[str, float | None]:
    """CO2_OPTIONS, restricted to co2default for v2+.

    global_mean_co2 is not an input channel in v2/v3 (see
    baseline_configs/versions.md), so masking it is meaningless there.
    """
    if version == "v1":
        return CO2_OPTIONS
    return {"co2default": CO2_OPTIONS["co2default"]}


HERE = pathlib.Path(__file__).parent
BASELINE_CONFIGS_DIR = HERE / "baseline_configs"
RUN_CONFIGS_DIR = HERE / "run_configs"

BASELINE_CONFIG_GLOB = "ace2-var-mask-nc-sfno-era5-v*.yaml"


def _discover_base_config_filenames() -> dict[str, str]:
    """Baseline config filename keyed by version, discovered from
    ``baseline_configs/`` (see ``versions.md`` there for what differs between
    versions). A version is only ever valid if a baseline file for it exists.
    """
    return {
        path.stem.rsplit("-", 1)[-1]: path.name
        for path in sorted(BASELINE_CONFIGS_DIR.glob(BASELINE_CONFIG_GLOB))
    }


# Baseline config sourced for every generated config, keyed by version, e.g.
# {"v1": "ace2-var-mask-nc-sfno-era5-v1.yaml", "v2": "...-v2.yaml"}.
BASE_CONFIG_FILENAMES = _discover_base_config_filenames()
DEFAULT_VERSION = "v1"


def stem_has_version(stem: str, version: str) -> bool:
    """True if a config stem is tagged with ``version`` (e.g. ``-v1``).

    Matches the version as a ``-``-delimited token so it works whether the
    tag is the final token (``...-v1``) or followed by another suffix
    (``...-v1-cooldown``, ``...-v1-bestinf``).
    """
    return bool(re.search(rf"(^|-){re.escape(version)}(-|$)", stem))


def config_name_to_run_name(name: str) -> str:
    """Wandb run name for a generated config stem (no .yaml).

    ``name`` already ends in ``-v1``/``-v2``/``-v3``, so no separate suffix is
    added.
    """
    suffix = name.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}"


def _fetch_wandb_run_names(project: str) -> set[str]:
    import wandb  # lazy import: only needed with --delete-if-in-wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{project}")
    return {run.name for run in runs}


def _apply_settings(
    cfg: dict, mask_level: int, co2_rate: float | None, keep_gmr: bool
) -> None:
    default: dict = {"max_masked_vars": mask_level}
    dropout: dict = {"default": default}
    if co2_rate is not None:
        dropout["override_groups"] = [
            {"variables": [CO2_FIELD], "masking": {"rate": co2_rate}}
        ]

    step_cfg = cfg["stepper"]["step"]["config"]
    step_cfg["input_dropout"] = dropout
    step_cfg["include_channel_mask_inputs"] = True
    if not keep_gmr:
        del step_cfg[GMR_FIELD]
    cfg["logging"]["project"] = WANDB_PROJECT


def _apply_spectral_band(cfg: dict, keep_band: bool) -> None:
    """Keep or drop the band-limited SFNO backbone knobs (``filter_num_
    groups``, ``spectral_ratio``) from the builder config. Dropping them
    falls back to the model's defaults (full-spectrum backbone, as in v3;
    see ``baseline_configs/versions.md``). No-op if ``keep_band`` (the
    baseline already has them, e.g. v4).
    """
    if keep_band:
        return
    builder_cfg = cfg["stepper"]["step"]["config"]["builder"]["config"]
    for field in SPECTRAL_BAND_FIELDS:
        builder_cfg.pop(field, None)


def iter_train_configs(version: str) -> list[tuple[str, dict]]:
    """``(name, config)`` for every masking training run of ``version``.

    Built in memory from the baseline config (no files written), so callers
    (e.g. generate_eval_configs.py) can enumerate every version's runs without
    the on-disk configs of that version being present.
    """
    base = yaml.safe_load(
        (BASELINE_CONFIGS_DIR / BASE_CONFIG_FILENAMES[version]).read_text()
    )
    configs: list[tuple[str, dict]] = []
    co2_options = co2_options_for_version(version)
    for mask_level in MASK_LEVELS:
        for gmr_name, keep_gmr in GMR_OPTIONS.items():
            for co2_name, co2_rate in co2_options.items():
                name = (
                    f"{BASE_CONFIG_STEM}-{gmr_name}-mask{mask_level}-{co2_name}"
                    f"-{version}"
                )
                cfg = copy.deepcopy(base)
                _apply_settings(cfg, mask_level, co2_rate, keep_gmr)
                configs.append((name, cfg))
    return configs


def _write_config(
    name: str,
    cfg: dict,
    wandb_run_names: set[str] | None = None,
) -> None:
    out_path = RUN_CONFIGS_DIR / f"{name}.yaml"
    if wandb_run_names is not None and config_name_to_run_name(name) in wandb_run_names:
        if out_path.exists():
            out_path.unlink()
            print(f"Deleted {out_path.name} (run exists in wandb)")
        else:
            print(f"Skipped {out_path.name} (run exists in wandb)")
        return
    out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"Wrote {out_path.name}")


def generate_configs(fetch_wandb: bool = False, version: str = DEFAULT_VERSION) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for yaml_path in RUN_CONFIGS_DIR.glob("*.yaml"):
        yaml_path.unlink()
        print(f"Removed {yaml_path.name}")

    wandb_run_names: set[str] | None = None
    if fetch_wandb:
        print(f"Fetching run names from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_run_names = _fetch_wandb_run_names(WANDB_PROJECT)
        print(f"Found {len(wandb_run_names)} existing runs.")

    for name, cfg in iter_train_configs(version):
        _write_config(name, cfg, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        "-v",
        choices=sorted(BASE_CONFIG_FILENAMES),
        default=DEFAULT_VERSION,
        help=f"Baseline config version to sweep from (default: {DEFAULT_VERSION}).",
    )
    parser.add_argument(
        "--delete-if-in-wandb",
        action="store_true",
        help=(
            "Skip (and delete any existing) configs whose run name already "
            "exists in the model's wandb project."
        ),
    )
    args = parser.parse_args()

    generate_configs(fetch_wandb=args.delete_if_in_wandb, version=args.version)


if __name__ == "__main__":
    main()
