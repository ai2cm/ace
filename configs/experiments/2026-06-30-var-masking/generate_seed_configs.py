"""Generate seed-replicate training configs from the nc-sfno era5 base config.

For a chosen subset of the var-masking sweep this writes ``n_seeds`` copies of
each config (default 5), differing only in the top-level ``seed`` field, so the
same masking scheme can be trained multiple times to estimate run-to-run spread.
Each generated config name ends in ``-v1``, ``-v2``, ``-v3`` or ``-v4``
(``--version`` selects which baseline config to source from, default v1; see
``baseline_configs/versions.md``). The ``global_mean_removal`` stepper config
is kept as in the baseline (GMR fixed on, no gmr token) for v1 and v4, unlike
``generate_masking_configs.py``; v2/v3/v5 sweep both gmron/gmroff (see
``gmr_options_for_version``).

Configs are written into ``run_configs/`` (only ``*-seed*.yaml`` files are
cleared first, leaving the other experiments' configs untouched). The masking
subset, each crossed with the co2 axis (co2default off / co2bern75 drops
``global_mean_co2`` w.p. 0.75) and with the seeds, is:

  - mask0:  no masking (``max_masked_vars`` = 0).
  - mask20: uniform 0-20 masking (``max_masked_vars`` = 20).

With the default 5 seeds this is 2 x 2 x 5 = 20 configs for v1:
mask20-co2bern75, mask20-co2default, mask0-co2bern75, mask0-co2default.
For v2/v3, global_mean_co2 is not an input channel (see
``baseline_configs/versions.md``), so the co2 axis is dropped, but the gmr
axis is added instead: 2 x 1 x 2 x 5 = 20 configs: mask20-gmron,
mask20-gmroff, mask0-gmron, mask0-gmroff (each co2default). v3 additionally
gets the sst axis and the clock50 arm described for v5 below, under the same
skip rules (sston skipped at mask0, clock50 GMR-on only); unlike v5, v3
config names keep the co2default token. That grows v3 to 20 + 10 (mask20
sston, both gmr options) + 5 (gmron mask20 clock50) = 35 configs.

For v4, global_mean_co2 is likewise not a native input and stays that way
(no co2-input axis: every v4 config has co2 excluded, matching the baseline),
and GMR is fixed on (no gmr axis, see ``gmr_options_for_version``). v4 also
drops the co2 token from the config name entirely (its single co2 option is
meaningless, see above). This is 2 x 1 x 5 = 10 configs: mask20, mask0 (each
GMR-on, co2default).

For v4 only, more arms are added on top of that sweep (``TARGETED_ARMS``),
each a targeted-masking config that pulls a named channel subset out of the
uniform pool into its own ``override_groups`` entry, dropped as a unit at a
fixed rate each step, while the remaining channels keep the ordinary
uniform-up-to-20 scheme (``max_masked_vars: 20``). This concentrates masking
budget on channels suspected of carrying the trend shortcut, instead of
spreading it thin over the full uniform pool:

- ``clock50``: the GMR global-mean channel
  (``__gmr_extra__surface_temperature``), rate 0.5.
- ``sst25``: ``surface_temperature`` + ``TMP2m`` (the classic trend-proxy
  pair), rate 0.25. Note ``surface_temperature`` is also the ocean module's
  prescribed field (``ocean.surface_temperature_name``); this arm doesn't
  special-case that interaction, only the GMR-guard one traced for clock50.

Each arm is fixed at GMR-on and v4's single co2 setting (co2default).

For v4 only, a co2-input axis (``co2_input_options_for_version``) compounds
with everything above — the mask0/mask20 sweep and every ``TARGETED_ARMS``
entry: ``co2in`` restores ``global_mean_co2`` as a network input
(``_apply_co2_input``, adding it to ``in_names`` + ``next_step_forcing_names``);
``co2out`` matches the v4 baseline (no co2 input). This doubles both: the mask
sweep becomes 2 x 2 x 5 = 20 configs, each targeted arm becomes 2 x 5 = 10
configs, for a v4 grand total of 20 + 10 * len(TARGETED_ARMS).

v5 is the mask0/mask20 sweep only (no ``TARGETED_ARMS``), crossed with the gmr
axis (gmron/gmroff) instead of the co2-input axis: every v5 config trains
without ``global_mean_co2`` as an input, with no co2 token in the config name.
v5 also adds an sst axis (``sst_options_for_version``): ``sston`` pins
``surface_temperature`` to never masked via a rate-0 Bernoulli
``override_groups`` entry (which also removes it from the uniform pool), while
the tokenless option keeps it in the uniform pool. ``sston`` is skipped at
mask0, where it would be a no-op. This is 2 (gmr) x 5 = 10 mask0 configs plus
2 (gmr) x 2 (sst) x 5 = 20 mask20 configs, i.e. 30:
gmron/gmroff x mask0, gmron/gmroff x mask20, gmron/gmroff x mask20-sston.

v5 also adds a ``clock50`` arm (from ``TARGETED_ARMS``): at mask20, the GMR
sentinel channel (``__gmr_extra__surface_temperature``) is pulled into its own
``override_groups`` entry and masked at rate 0.5 on top of the uniform pool.
It is GMR-on only — ``gmroff`` removes ``global_mean_removal``, so the sentinel
channel is never packed and the arm is impossible there (masking it would fail
config validation, cf. the ``sston``-at-mask0 skip) — and is not crossed with
the sst axis. This adds 1 (gmron) x 5 = 5 configs (gmron x mask20-clock50), a
v5 grand total of 35.
"""

import argparse
import copy
from collections.abc import Sequence
from typing import NamedTuple

import yaml
from generate_masking_configs import (
    BASE_CONFIG_FILENAMES,
    BASE_CONFIG_STEM,
    BASELINE_CONFIGS_DIR,
    DEFAULT_VERSION,
    GMR_OPTIONS,
    RUN_CONFIGS_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    _apply_co2_input,
    _apply_settings,
    _fetch_wandb_run_names,
    co2_options_for_version,
    config_name_to_run_name,
)

DEFAULT_N_SEEDS = 5


def gmr_options_for_version(version: str) -> dict[str, bool]:
    """GMR_OPTIONS, restricted to keep-baseline-only (GMR fixed on) for v1/v4.

    No gmr axis for v1 or v4 (``global_mean_removal`` is kept on, as in the
    baseline, with no gmr token in the config name); v2/v3/v5 sweep both
    gmron/gmroff, as in ``generate_masking_configs.py``.
    """
    if version in ("v1", "v4"):
        return {"": True}
    return GMR_OPTIONS


# Whether to re-add global_mean_co2 as a network input, keyed by config/job
# name token.
CO2_INPUT_OPTIONS = {"co2in": True, "co2out": False}


def co2_input_options_for_version(version: str) -> dict[str, bool]:
    """CO2_INPUT_OPTIONS, restricted to a no-op single option except for v4.

    v4's baseline drops ``global_mean_co2`` as an input (see
    ``baseline_configs/versions.md``); this axis re-adds it for an ablation
    (``co2in``) alongside the baseline (``co2out``), compounding with the
    mask sweep and every ``TARGETED_ARMS`` entry. Other versions keep the
    config name and network input unchanged, via a single tokenless no-op
    option matching their baseline: co2 is an input only in v1 (v2/v3/v5
    drop it, see ``versions.md``), so v1 gets ``{"": True}`` and the rest
    ``{"": False}`` — returning True for v2/v3 would silently re-add
    ``global_mean_co2`` as an input via ``_apply_co2_input``.
    """
    if version == "v4":
        return CO2_INPUT_OPTIONS
    if version == "v1":
        return {"": True}
    return {"": False}


# The prescribed-SST input channel (also the ocean module's
# ocean.surface_temperature_name).
SST_FIELD = "surface_temperature"

# Bernoulli rate for an override_groups entry pulling SST out of the uniform
# pool, keyed by config/job name token. ``sston``: rate 0, SST never masked.
SST_OPTIONS: dict[str, float | None] = {"": None, "sston": 0.0}


def sst_options_for_version(version: str) -> dict[str, float | None]:
    """SST_OPTIONS, restricted to the no-op (SST in uniform pool) except v3/v5.

    v3/v5 add an ``sston`` ablation: ``surface_temperature`` is moved into its
    own ``override_groups`` entry with Bernoulli rate 0, so it is never masked
    (a rate-0 Bernoulli group never fires, and grouped channels leave the
    uniform pool). The tokenless option keeps SST in the uniform pool, as in
    the baseline. Redundant combinations (``sston`` at mask0, where nothing is
    masked anyway) are skipped in ``iter_train_configs``.
    """
    if version in ("v3", "v5"):
        return SST_OPTIONS
    return {"": None}


class SeedGroup(NamedTuple):
    label: str
    mask_level: int


# Masking subset to replicate across seeds.
SEED_GROUPS = [
    SeedGroup("mask0", 0),
    SeedGroup("mask20", 20),
]


class TargetedArm(NamedTuple):
    label: str
    channels: Sequence[str]
    rate: float


TARGETED_MASK_LEVEL = 20  # uniform max_masked_vars for the non-targeted channels

# The GMR-sentinel targeted arm (masks the shared global-mean guard channel,
# __gmr_extra__surface_temperature, at rate 0.5 on top of the uniform pool).
# Used by v4's TARGETED_ARMS and, GMR-on only, by v5 (see iter_train_configs).
CLOCK_ARM = TargetedArm("clock50", ["__gmr_extra__surface_temperature"], 0.5)

# v4-only targeted-masking arms: each pulls its channels out of the uniform
# pool into their own override_groups entry (see TARGETED_MASK_LEVEL).
TARGETED_ARMS = [
    CLOCK_ARM,
    TargetedArm("sst25", ["surface_temperature", "TMP2m"], 0.25),
]


def iter_train_configs(
    version: str, n_seeds: int = DEFAULT_N_SEEDS
) -> list[tuple[str, dict]]:
    """``(name, config)`` for every seed-replicate training run of ``version``.

    Built in memory from the baseline config (no files written), so callers
    (e.g. generate_eval_configs.py) can enumerate every version's runs without
    the on-disk configs of that version being present.
    """
    base = yaml.safe_load(
        (BASELINE_CONFIGS_DIR / BASE_CONFIG_FILENAMES[version]).read_text()
    )
    configs: list[tuple[str, dict]] = []
    co2_options = co2_options_for_version(version)
    gmr_options = gmr_options_for_version(version)
    co2_input_options = co2_input_options_for_version(version)
    sst_options = sst_options_for_version(version)
    for group in SEED_GROUPS:
        for co2_name, co2_rate in co2_options.items():
            for gmr_name, keep_gmr in gmr_options.items():
                for co2_input_name, include_co2_input in co2_input_options.items():
                    for sst_name, sst_rate in sst_options.items():
                        if sst_rate is not None and group.mask_level == 0:
                            # mask0 masks nothing, so pulling SST out of the
                            # uniform pool is a no-op; skip the duplicate run.
                            continue
                        gmr_token = f"{gmr_name}-" if gmr_name else ""
                        co2_input_token = f"{co2_input_name}-" if co2_input_name else ""
                        # v4/v5 have only one (meaningless) co2 option, drop
                        # token.
                        co2_token = "" if version in ("v4", "v5") else f"-{co2_name}"
                        sst_token = f"-{sst_name}" if sst_name else ""
                        base_name = (
                            f"{BASE_CONFIG_STEM}-{gmr_token}{co2_input_token}"
                            f"{group.label}{co2_token}{sst_token}"
                        )
                        extra_override_groups = None
                        if sst_rate is not None:
                            extra_override_groups = [
                                {
                                    "variables": [SST_FIELD],
                                    "masking": {"rate": sst_rate},
                                }
                            ]
                        for seed in range(n_seeds):
                            name = f"{base_name}-seed{seed}-{version}"
                            cfg = copy.deepcopy(base)
                            _apply_co2_input(cfg, include_co2_input)
                            _apply_settings(
                                cfg,
                                group.mask_level,
                                co2_rate,
                                keep_gmr,
                                extra_override_groups=extra_override_groups,
                            )
                            cfg["seed"] = seed
                            configs.append((name, cfg))

    if version == "v4":
        # Targeted arms: fixed GMR-on, v4's single (co2default) co2 setting,
        # crossed with the co2-input axis; not crossed with the sweep above
        # or each other (see module docstring).
        _co2_name, co2_rate = next(iter(co2_options.items()))
        for arm in TARGETED_ARMS:
            for co2_input_name, include_co2_input in co2_input_options.items():
                co2_input_token = f"{co2_input_name}-" if co2_input_name else ""
                base_name = (
                    f"{BASE_CONFIG_STEM}-{co2_input_token}{arm.label}"
                    f"-mask{TARGETED_MASK_LEVEL}"
                )
                for seed in range(n_seeds):
                    name = f"{base_name}-seed{seed}-{version}"
                    cfg = copy.deepcopy(base)
                    _apply_co2_input(cfg, include_co2_input)
                    _apply_settings(
                        cfg,
                        TARGETED_MASK_LEVEL,
                        co2_rate,
                        keep_gmr=True,
                        extra_override_groups=[
                            {
                                "variables": list(arm.channels),
                                "masking": {"rate": arm.rate},
                            }
                        ],
                    )
                    cfg["seed"] = seed
                    configs.append((name, cfg))

    if version in ("v3", "v5"):
        # v3/v5 clock50 arm: on top of the uniform mask20 pool, pull the GMR
        # sentinel channel (__gmr_extra__surface_temperature) into its own
        # group masked at rate 0.5. GMR-on only: gmroff removes
        # global_mean_removal, so the sentinel channel is never packed and the
        # arm is impossible there (cf. the sston-at-mask0 skip above). Not
        # crossed with the sst axis. Neither version has a co2 input; v3
        # keeps its co2default name token (v5 has none, see co2_token above).
        gmron_name = next(name for name, keep in GMR_OPTIONS.items() if keep)
        co2_token = "" if version == "v5" else f"-{next(iter(co2_options))}"
        for seed in range(n_seeds):
            name = (
                f"{BASE_CONFIG_STEM}-{gmron_name}-mask{TARGETED_MASK_LEVEL}"
                f"-{CLOCK_ARM.label}{co2_token}-seed{seed}-{version}"
            )
            cfg = copy.deepcopy(base)
            _apply_co2_input(cfg, include_co2=False)
            _apply_settings(
                cfg,
                TARGETED_MASK_LEVEL,
                co2_rate=None,
                keep_gmr=True,
                extra_override_groups=[
                    {
                        "variables": list(CLOCK_ARM.channels),
                        "masking": {"rate": CLOCK_ARM.rate},
                    }
                ],
            )
            cfg["seed"] = seed
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


def generate_configs(
    n_seeds: int = DEFAULT_N_SEEDS,
    fetch_wandb: bool = False,
    version: str = DEFAULT_VERSION,
) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for yaml_path in RUN_CONFIGS_DIR.glob("*-seed*.yaml"):
        yaml_path.unlink()
        print(f"Removed {yaml_path.name}")

    wandb_run_names: set[str] | None = None
    if fetch_wandb:
        print(f"Fetching run names from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_run_names = _fetch_wandb_run_names(WANDB_PROJECT)
        print(f"Found {len(wandb_run_names)} existing runs.")

    for name, cfg in iter_train_configs(version, n_seeds):
        _write_config(name, cfg, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        dest="n_seeds",
        help=f"Number of seeds per config (default: {DEFAULT_N_SEEDS}).",
    )
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

    generate_configs(
        args.n_seeds, fetch_wandb=args.delete_if_in_wandb, version=args.version
    )


if __name__ == "__main__":
    main()
