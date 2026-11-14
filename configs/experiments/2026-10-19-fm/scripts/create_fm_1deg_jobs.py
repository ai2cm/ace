"""Generate the 1deg, 6-hourly FM run configs (wandb group
ace2-fm-1deg-2026-11-13).

6 runs: nc-sfno-v2 x 2 data arms (fm-e3sm-balanced, fm-era5) x 3 seeds, named
ace-train-config-1deg-nc-sfno-v2-{arm}-s{seed}.yaml. Each is its hand-made
parent base_configs/ace-train-config-1deg-nc-sfno-v2-{arm}.yaml (see its header
for the recipe) with only `seed` changed.

Plus one smoke-test config per arm, ...-s0-smoke.yaml: seed 0 with the changes
below, to prove at 1deg on the production GPU count that training at the
longest rollout fits in memory, that validation, inline inference and
checkpointing work, and to measure throughput, in about an hour instead of
days. Everything else (data, normalization, model, batch size, time buffer and
pool, optimizer, LR warmup, EMA, EMA checkpoints every epoch) is the parent's.
- max_epochs 2 and label_sampling samples_per_epoch 256
  (SMOKE_SAMPLES_PER_EPOCH): 288 iterations per epoch at batch_size 8 and
  time_buffer 8 (32 window batches x 9), against 12420 in the real runs. 256 is
  the smallest epoch that fills the time_buffer_pool_size 32 pool of windows
  held on each GPU, as the real runs do.
- Step schedule: 1 forward step in epoch 1, then in epoch 2 the parent's
  sampled schedule with its outcomes (1, 2, 4, 12, 20 steps, loss on the last
  step) reweighted to {1: .2, 2: .1, 4: .1, 12: .1, 20: .5}
  (SMOKE_STEP_PROBABILITIES), so about half of the 288 batches, instead of 5%,
  are 20-step rollouts. The mean is 12 steps per batch (3 in the real runs).
- log_train_every_n_batches 20 (default 100), so each phase logs
  training_samples_per_second_on_rank_0 14 times; epoch_train_seconds is
  logged per epoch as usual.
- Validation: every entry of the parent (val, and val_c96 for
  fm-e3sm-balanced) kept, each year (1994, 2014) cut to January-March
  (SMOKE_VALIDATION_STOP): 17 batches of 32 per entry.
- Inline inference, both every epoch: aimip_checkpoint_era5 (weight 1, the
  best_inference_ckpt selection) with n_forward_steps 400 (100 days; 7300 in
  the parent) and weather_2020_era5 (weight 0) as in the parent (20 steps, 4
  members x 8 ICs). The parent's 10year_oosample_varying_co2_era5,
  long_46year_varying_co2_era5 and weather_1950_era5 are dropped.

The configs are written by a YAML round trip, so the parents' comments are
not carried over; the header of each file points back to its parent.

Usage:
    python create_fm_1deg_jobs.py
"""

import copy
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
EXP_DIR = HERE.parent  # experiment root (configs, records); scripts live in scripts/
BASE_CONFIGS_DIR = EXP_DIR / "base_configs"
RUN_CONFIGS_DIR = EXP_DIR / "run_configs"
CONFIG_PREFIX = "ace-train-config-1deg-"

ARCH = "nc-sfno-v2"
ARMS = ("fm-e3sm-balanced", "fm-era5")
SEEDS = (0, 1, 2)

SMOKE_SEED = 0
SMOKE_MAX_EPOCHS = 2
SMOKE_SAMPLES_PER_EPOCH = 256
# forward steps -> probability, for the outcomes of the parent's schedule
SMOKE_STEP_PROBABILITIES = {1: 0.2, 2: 0.1, 4: 0.1, 12: 0.1, 20: 0.5}
SMOKE_LOG_TRAIN_EVERY_N_BATCHES = 20
VALIDATION_YEARS = (1994, 2014)
SMOKE_VALIDATION_STOP = "03-31"
SELECTION_INFERENCE = "aimip_checkpoint_era5"
SMOKE_SELECTION_N_FORWARD_STEPS = 400
SMOKE_WEATHER_INFERENCE = "weather_2020_era5"
EVERY_EPOCH = {"start": 0, "step": 1}


def parent_path(arm: str) -> pathlib.Path:
    return BASE_CONFIGS_DIR / f"{CONFIG_PREFIX}{ARCH}-{arm}.yaml"


def run_config_name(arm: str, seed: int, smoke: bool = False) -> str:
    return f"{CONFIG_PREFIX}{ARCH}-{arm}-s{seed}{'-smoke' if smoke else ''}.yaml"


def make_config(arm: str, seed: int) -> dict:
    cfg = yaml.safe_load(parent_path(arm).read_text())
    assert cfg["seed"] == 0
    cfg["seed"] = seed
    return cfg


def _smoke_validation(parent_validation: list[dict]) -> list[dict]:
    validation = copy.deepcopy(parent_validation)
    for entry in validation:
        concat = entry["loader"]["dataset"]["concat"]
        assert len(concat) == len(VALIDATION_YEARS)
        for dataset, year in zip(concat, VALIDATION_YEARS):
            assert dataset["subset"] == {
                "start_time": f"{year}-01-01",
                "stop_time": f"{year}-12-31",
            }
            dataset["subset"]["stop_time"] = f"{year}-{SMOKE_VALIDATION_STOP}"
    return validation


def _smoke_inference(parent_inference: list[dict]) -> list[dict]:
    by_name = {entry["name"]: entry for entry in parent_inference}
    selection = copy.deepcopy(by_name[SELECTION_INFERENCE])
    assert selection["weight"] == 1.0
    assert selection["epochs"] == EVERY_EPOCH
    assert selection["n_forward_steps"] > SMOKE_SELECTION_N_FORWARD_STEPS
    selection["n_forward_steps"] = SMOKE_SELECTION_N_FORWARD_STEPS
    weather = copy.deepcopy(by_name[SMOKE_WEATHER_INFERENCE])
    assert weather["weight"] == 0.0
    weather["epochs"] = dict(EVERY_EPOCH)
    return [selection, weather]


def make_smoke_config(arm: str) -> dict:
    cfg = make_config(arm, SMOKE_SEED)
    cfg["max_epochs"] = SMOKE_MAX_EPOCHS
    assert "log_train_every_n_batches" not in cfg
    cfg["log_train_every_n_batches"] = SMOKE_LOG_TRAIN_EVERY_N_BATCHES
    assert cfg["ema_checkpoint_save_epochs"] == {"start": 1, "step": 1}

    train_loader = cfg["train_loader"]
    # 288 iterations per epoch and a full pool, as the module docstring says.
    assert train_loader["batch_size"] == 8
    assert train_loader["time_buffer"] == 8
    assert train_loader["time_buffer_pool_size"] * 8 == SMOKE_SAMPLES_PER_EPOCH
    train_loader["label_sampling"]["samples_per_epoch"] = SMOKE_SAMPLES_PER_EPOCH

    schedule = cfg["stepper_training"]["n_forward_steps"]
    assert schedule["start_value"] == 1
    (milestone,) = schedule["milestones"]
    outcomes = milestone["value"]["outcomes"]
    assert [o["steps"] for o in outcomes] == list(SMOKE_STEP_PROBABILITIES)
    milestone["epoch"] = SMOKE_MAX_EPOCHS
    for outcome in outcomes:
        outcome["probability"] = SMOKE_STEP_PROBABILITIES[outcome["steps"]]

    cfg["validation"] = _smoke_validation(cfg["validation"])
    cfg["inference"] = _smoke_inference(cfg["inference"])
    return cfg


COMPILE_MODE = "default"  # torch.compile inductor mode, see header()


def header(arm: str, smoke: bool) -> str:
    lines = [
        "# Generated by create_fm_1deg_jobs.py; do not edit by hand.",
        f"# Parent: base_configs/{parent_path(arm).name}",
    ]
    if smoke:
        lines.append(
            "# Smoke test; see create_fm_1deg_jobs.py for every change from the parent."
        )
    else:
        lines.append("# Differs from the parent only in `seed`.")
    lines += [
        "# Compiled without CUDA graphs: at 1deg their memory pools (54 GiB of an",
        "# 80 GiB H100 after one epoch) left no room for validation, and the kernels",
        "# are too large for the launch overhead CUDA graphs save to matter.",
        f"# arg: --env FME_COMPILE_MODE={COMPILE_MODE}",
    ]
    return "\n".join(lines) + "\n"


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=1000)


def _write(arm: str, seed: int, smoke: bool, cfg: dict) -> None:
    path = RUN_CONFIGS_DIR / run_config_name(arm, seed, smoke)
    path.write_text(header(arm, smoke) + dump(cfg))
    print(f"Wrote {path.relative_to(EXP_DIR)}")


def main() -> None:
    for arm in ARMS:
        for seed in SEEDS:
            _write(arm, seed, False, make_config(arm, seed))
        _write(arm, SMOKE_SEED, True, make_smoke_config(arm))


if __name__ == "__main__":
    main()
