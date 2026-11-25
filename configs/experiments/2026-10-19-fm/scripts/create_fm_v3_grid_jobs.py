"""Generate the 2026-11-25 FM v3 grid runs (wandb group ace2-fm-v3-grid-2026-11-25).

Every run uses the FM v3 recipe of create_fm_v3_jobs.py (120 epochs from
scratch, LR 1e-4 constant then 3-step + 1-sqrt cooldown to 0 over epochs
115-120, 3105 iterations per epoch, per-grid-cell network normalization and
the `cellloss` per-cell loss normalization). Two grids:

A. 5-source grid (21 runs), ace2-fm-{arch}-fm-e3sm-{mix}-c2-cellloss-s{seed}-
   {mask}: arch {nc-sfno-v3, nc-swin-v2.2-v3} x mix {balanced, temperature} x
   mask {maskon, maskoff} x seeds {0, 1, 2}, without the 3 nc-sfno-v3 /
   balanced / maskoff cells, which are the finished create_fm_v3_jobs.py runs
   ace2-fm-nc-sfno-v3-fm-e3sm-balanced-c2-cellloss-s{0,1,2}-maskoff (the
   "reference"). Each config is the reference config of the same seed
   (create_fm_v3_jobs.make_config, network c2, loss cellloss, STATS_DIR) with
   only these changes:
   - temperature: label_sampling weights w_i = n_i^ALPHA / sum_j n_j^ALPHA
     with ALPHA = 0.7 and n_i the training timesteps of label i
     (create_fm_v3_jobs.C3_TIMESTEPS summed per label: c96 83296, era5 24836,
     cm4 141353, xshield 4020, e3sm 38690), i.e. c96 0.27924, era5 0.119701,
     cm4 0.404348, xshield 0.033458, e3sm 0.163253 (rounded to 6 decimals;
     fme normalizes the weights). balanced keeps the reference's 0.2 each.
     samples_per_epoch stays 2758 (3105 iterations per epoch). The c2 stats
     are the reference's (their root pools the labels equally, as the
     balanced sampler does; the grouped per-label stats do not depend on the
     mix).
   - maskon: `input_dropout: {default: {max_masked_vars: 10}}` in the step
     config, as the v2 -maskon runs (create_fm_v2_jobs.py) add it; maskoff
     has none (as the reference).
   - nc-swin-v2.2-v3: the model and optimizer of the v2 Swin fm-e3sm-balanced
     maskoff run of the same seed (run_configs/ace-train-config-4deg-nc-swin-
     v2.2-v2-fm-e3sm-balanced-s{seed}-maskoff.yaml): the step config's
     `builder` (NoiseConditionedSwinTransformer, embed_dim 256) and
     `float32_matmul_precision: high`, and the optimization block (FusedAdam,
     weight decay 0.01, LR 1e-4, max_consecutive_non_finite_losses 5); the
     generator asserts that the v2 Swin and SFNO configs differ only there
     (and in the scheduler). The scheduler is the v3 one with the Swin
     parent's 2-epoch warmup kept, as per-iteration ConstantLR pieces of one
     epoch each at factors 0.01 and 0.505 (exactly as
     create_fm_c96amip_jobs.py does for Swin: lr_scheduler("parent", ...)).
     EMA 0.9999, normalization and loss as the SFNO reference.

B. Single-source grid (12 runs), ace2-fm-{arch}-fm-{c96amip,era5-1979}-c1-
   cellloss-s{seed}-maskoff: arch {nc-sfno-v3, nc-swin-v2.2-v3} x source x
   seeds {0, 1, 2}. Each config is the create_fm_c96amip_jobs.py config of the
   same arch (nc-sfno-v2 / nc-swin-v2.2-v2, already on the v3 recipe, Swin
   with its 2-epoch warmup), arm and seed, i.e. exactly its data (C96 AMIP
   ic_0001 or ERA5, 1979-1993 + 1995-2008, the single label at weight 1.0),
   validation and inline inference selection, with its scalar normalization
   replaced by the v3 c1-cellloss normalization
   (create_fm_v3_jobs.make_normalization(..., "c1", "cellloss", stats_dir)):
   network = pooled per-cell (floor 0.01), loss and residual = pooled per-cell
   (floor 0.1), no grouped block (one source: c2 == c1). The per-cell stats
   come from the jobs `submit_fm_v3_jobs.py --stats-jobs` submits
   (scripts/data_process/configs/fm-v3-{c96amip,era5-1979}-percell-stats.yaml;
   the five scalar-kept variables equal the arm's scalar stats); their
   output directories are known only once those jobs are submitted, so the
   single-source configs are written only with --single-source-stats.

Each file starts with the `# KEY: value` header of create_fm_v3_jobs.py
(create_anneal_jobs.read_header). The configs keep `experiment_dir:
/results`; submit_fm_v3_jobs.py overrides it with $AGENT_OUTPUT_DIR.

Usage:
    python create_fm_v3_grid_jobs.py
        [--single-source-stats c96amip=DIR era5-1979=DIR] [--out-dir DIR]
"""

import argparse
import copy
import pathlib
from dataclasses import dataclass

import create_fm_c96amip_jobs as c96amip
import create_fm_v3_jobs as v3
import yaml
from create_fm_v2_followup_jobs import (
    ITERATIONS_PER_EPOCH,
    PARENT_WARMUP_FACTORS,
    check_iterations_per_epoch,
    lr_scheduler,
)
from create_fm_v2_jobs import (
    CONFIG_PREFIX,
    MAX_MASKED_VARS,
    RUN_CONFIGS_DIR,
    SAMPLES_PER_EPOCH,
    SEEDS,
    run_config_name,
)

EXP_DIR = RUN_CONFIGS_DIR.parent
REPO_ROOT = EXP_DIR.parents[2]
JOB_PREFIX = "ace2-fm-"
WANDB_GROUP = "ace2-fm-v3-grid-2026-11-25"

SFNO = "nc-sfno-v3"
SWIN = "nc-swin-v2.2-v3"
ARCHS = (SFNO, SWIN)
# The v2 architecture each derives from.
V2_ARCHS = {SFNO: c96amip.SFNO, SWIN: c96amip.SWIN}
SWIN_EMBED_DIM = 256

MULTI_SOURCE = "e3sm"  # the fm-e3sm-balanced data of the reference
SINGLE_SOURCES = ("c96amip", "era5-1979")
SINGLE_SOURCE_ARMS = {"c96amip": c96amip.ARM, "era5-1979": c96amip.ERA5_ARM}
MIXES = ("balanced", "temperature")
MASKS = (False, True)
NETWORKS = {MULTI_SOURCE: "c2", **{source: "c1" for source in SINGLE_SOURCES}}
LOSS = "cellloss"

# Temperature-scaled label sampling (5-source grid).
ALPHA = 0.7
LABEL_TIMESTEPS = {
    label: sum(
        v3.C3_TIMESTEPS[child]
        for child in v3.C3_SPLITS.get(label, {label: label}).values()
    )
    for label in v3.GROUP_LABELS
}
assert LABEL_TIMESTEPS == {
    "c96": 83296,
    "era5": 24836,
    "cm4": 141353,
    "xshield": 4020,
    "e3sm": 38690,
}
assert v3.iterations_per_epoch(SAMPLES_PER_EPOCH) == ITERATIONS_PER_EPOCH == 3105


def temperature_weights(
    timesteps: dict[str, int] = LABEL_TIMESTEPS, alpha: float = ALPHA
) -> dict[str, float]:
    total = sum(n**alpha for n in timesteps.values())
    return {label: round(n**alpha / total, 6) for label, n in timesteps.items()}


@dataclass(frozen=True)
class Run:
    arch: str
    source: str
    seed: int
    mix: str = "balanced"
    mask: bool = False
    smoke = False  # submit_fm_v3_jobs.py reads run.smoke

    @property
    def network(self) -> str:
        return NETWORKS[self.source]

    @property
    def stem(self) -> str:
        mix = f"-{self.mix}" if self.source == MULTI_SOURCE else ""
        mask = "maskon" if self.mask else "maskoff"
        return (
            f"{self.arch}-fm-{self.source}{mix}-{self.network}-{LOSS}-"
            f"s{self.seed}-{mask}"
        )

    @property
    def config_name(self) -> str:
        return f"{CONFIG_PREFIX}{self.stem}.yaml"

    @property
    def job_name(self) -> str:
        # Same as submit_fm_jobs.config_to_job_name(self.config_name).
        return f"{JOB_PREFIX}{self.stem}"

    @property
    def path(self) -> pathlib.Path:
        return RUN_CONFIGS_DIR / self.config_name

    @property
    def group(self) -> str:
        return WANDB_GROUP

    @property
    def parent_config(self) -> pathlib.Path:
        """The v2 run config the model and optimizer come from."""
        if self.source == MULTI_SOURCE:
            return RUN_CONFIGS_DIR / run_config_name(
                V2_ARCHS[self.arch], v3.ARM, self.seed, False
            )
        return self._c96amip_run().parent_config

    def _c96amip_run(self) -> c96amip.Run:
        return c96amip.Run(
            V2_ARCHS[self.arch], self.seed, arm=SINGLE_SOURCE_ARMS[self.source]
        )

    def _reference_run(self) -> v3.Run:
        return v3.Run(self.network, LOSS, self.seed)


def is_reference(run: Run) -> bool:
    """The finished create_fm_v3_jobs.py runs (same stem)."""
    return (
        run.arch == SFNO
        and run.source == MULTI_SOURCE
        and run.mix == "balanced"
        and not run.mask
    )


def multi_source_runs(include_reference: bool = False) -> list[Run]:
    all_runs = [
        Run(arch, MULTI_SOURCE, seed, mix, mask)
        for arch in ARCHS
        for mix in MIXES
        for mask in MASKS
        for seed in SEEDS
    ]
    return [r for r in all_runs if include_reference or not is_reference(r)]


def single_source_runs() -> list[Run]:
    return [
        Run(arch, source, seed)
        for source in SINGLE_SOURCES
        for arch in ARCHS
        for seed in SEEDS
    ]


# --- per-cell stats of the single sources ------------------------------------------


@dataclass(frozen=True)
class StatsJob:
    """A per-cell stats job (submit_fm_v3_jobs.py --stats-jobs): writes
    $AGENT_OUTPUT_DIR/<output_name> with get_pooled_stats.py and checks it
    with verify_percell_stats.py against `train_config`, the scalar-normalized
    fm-<source> run config, whose training data the grid runs keep.
    """

    source: str

    @property
    def job_name(self) -> str:
        return f"fm-v3-{self.source}-percell-stats"

    @property
    def output_name(self) -> str:
        return f"2026-11-25-fm-v3-{self.source}-percell"

    @property
    def stats_config(self) -> pathlib.Path:
        return (
            REPO_ROOT
            / "scripts/data_process/configs"
            / f"fm-v3-{self.source}-percell-stats.yaml"
        )

    @property
    def train_config(self) -> pathlib.Path:
        return c96amip.Run(c96amip.SFNO, 0, arm=SINGLE_SOURCE_ARMS[self.source]).path

    @property
    def description(self) -> str:
        return (
            f"Per-grid-cell stats ({self.stats_config.name}) for the FM v3 grid "
            f"{self.source} runs, on a GPU node"
        )


def stats_jobs() -> list[StatsJob]:
    return [StatsJob(source) for source in SINGLE_SOURCES]


# --- configs ---------------------------------------------------------------------

SWIN_STEP_KEYS = ("builder", "float32_matmul_precision")
SWIN_OPTIMIZATION_KEYS = ("scheduler", "max_consecutive_non_finite_losses")


def _without_swin_fields(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for key in SWIN_OPTIMIZATION_KEYS:
        cfg["optimization"].pop(key, None)
    step_config = cfg["stepper"]["step"]["config"]
    for key in SWIN_STEP_KEYS:
        step_config.pop(key, None)
    return cfg


def _to_swin(cfg: dict, seed: int) -> dict:
    """The SFNO v3 config `cfg` with the v2 Swin model and optimizer."""
    sfno_parent = yaml.safe_load(
        (
            RUN_CONFIGS_DIR / run_config_name(c96amip.SFNO, v3.ARM, seed, False)
        ).read_text()
    )
    swin_parent = yaml.safe_load(
        (
            RUN_CONFIGS_DIR / run_config_name(c96amip.SWIN, v3.ARM, seed, False)
        ).read_text()
    )
    # The v2 parents differ only in the Swin fields.
    assert _without_swin_fields(swin_parent) == _without_swin_fields(sfno_parent)
    swin_step = swin_parent["stepper"]["step"]["config"]
    assert swin_step["builder"]["type"] == "NoiseConditionedSwinTransformer"
    assert swin_step["builder"]["config"]["embed_dim"] == SWIN_EMBED_DIM
    assert swin_step["float32_matmul_precision"] == "high"
    swin_optimization = swin_parent["optimization"]
    assert swin_optimization["max_consecutive_non_finite_losses"] == 5
    assert swin_optimization["scheduler"]["schedulers"][0] == {
        "type": "LinearLR",
        "kwargs": {"start_factor": 0.01, "end_factor": 1.0, "total_iters": 2},
        "step_each_iteration": False,
    }
    assert PARENT_WARMUP_FACTORS == (0.01, 0.505)

    cfg = copy.deepcopy(cfg)
    # Keys in the Swin parent's order; values from the SFNO v3 config except
    # the Swin fields.
    optimization = copy.deepcopy(swin_optimization)
    for key, value in cfg["optimization"].items():
        if key not in SWIN_OPTIMIZATION_KEYS:
            assert optimization[key] == value, key
    optimization["scheduler"] = lr_scheduler(
        "parent", v3.MAX_EPOCHS, v3.COOLDOWN_EPOCHS
    )
    cfg["optimization"] = optimization
    sfno_step = cfg["stepper"]["step"]["config"]
    assert set(swin_step) == set(sfno_step) | {"float32_matmul_precision"}
    cfg["stepper"]["step"]["config"] = {
        key: copy.deepcopy(swin_step[key] if key in SWIN_STEP_KEYS else sfno_step[key])
        for key in swin_step
    }
    return cfg


def make_multi_source_config(run: Run) -> dict:
    cfg = v3.make_config(run._reference_run(), v3.STATS_DIR)
    if run.arch == SWIN:
        cfg = _to_swin(cfg, run.seed)
    sampling = cfg["train_loader"]["label_sampling"]
    assert sampling == {
        "weights": {label: 0.2 for label in v3.GROUP_LABELS},
        "samples_per_epoch": SAMPLES_PER_EPOCH,
    }
    if run.mix == "temperature":
        sampling["weights"] = temperature_weights()
    elif run.mix != "balanced":
        raise ValueError(run.mix)
    step_config = cfg["stepper"]["step"]["config"]
    assert "input_dropout" not in step_config
    if run.mask:
        step_config["input_dropout"] = {"default": {"max_masked_vars": MAX_MASKED_VARS}}
    return cfg


def make_single_source_config(run: Run, stats_dir: str) -> dict:
    # The scalar stats dir is replaced below; any path builds the config.
    c96amip_run = run._c96amip_run()
    cfg = c96amip.make_config(c96amip_run, c96amip.STATS_DIRS[c96amip_run.arm])
    # The v3 c1-cellloss normalization (which does not depend on the parent
    # normalization make_normalization checks).
    e3sm_parent = yaml.safe_load(run._reference_run().parent_config.read_text())
    cfg["stepper"]["step"]["config"]["normalization"] = v3.make_normalization(
        e3sm_parent["stepper"]["step"]["config"]["normalization"],
        run.network,
        LOSS,
        stats_dir,
    )
    return cfg


def make_config(run: Run, single_source_stats: dict[str, str] | None = None) -> dict:
    if run.source == MULTI_SOURCE:
        cfg = make_multi_source_config(run)
    else:
        if not single_source_stats or run.source not in single_source_stats:
            raise ValueError(f"{run.job_name}: no stats dir for {run.source}")
        cfg = make_single_source_config(run, single_source_stats[run.source])
    check_iterations_per_epoch(cfg)
    assert cfg["seed"] == run.seed
    return cfg


def describe(run: Run) -> str:
    if run.source == MULTI_SOURCE:
        data = f"5 sources, {run.mix} label sampling"
        if run.mix == "temperature":
            data += f" (alpha {ALPHA})"
        norm = "c2 (pooled + grouped per-cell) network normalization"
    else:
        data = {
            "c96amip": "C96 AMIP ic_0001 only",
            "era5-1979": "ERA5 1979-2008 only",
        }[run.source]
        norm = "c1 (pooled per-cell) network normalization"
    warmup = ", 2-epoch warmup" if run.arch == SWIN else ""
    mask = "masking on" if run.mask else "masking off"
    return (
        f"ACE FM v3 grid {run.arch} on {data}: {v3.MAX_EPOCHS} epochs from "
        f"scratch, LR 1e-4 constant{warmup}, 3-step + 1-sqrt cooldown over "
        f"epochs {v3.MULTI_STEP_FROM}-{v3.MAX_EPOCHS}, {norm}, per-cell loss "
        f"normalization, {mask}"
    )


def header(run: Run) -> str:
    lines = [
        "Generated by create_fm_v3_grid_jobs.py; do not edit by hand.",
        f"JOB_NAME: {run.job_name}",
        f"WANDB_GROUP: {run.group}",
        f"PARENT_CONFIG: run_configs/{run.parent_config.name}",
        f"WEKA_MOUNT: {v3.ERA5_STORE}",
        "EXPERIMENT_DIR: $AGENT_OUTPUT_DIR (set by submit_fm_v3_jobs.py)",
        f"DESCRIPTION: {describe(run)}",
        "See create_fm_v3_grid_jobs.py for every change from the parent.",
    ]
    return "".join(f"# {line}\n" for line in lines)


def render(run: Run, single_source_stats: dict[str, str] | None = None) -> str:
    body = yaml.safe_dump(
        make_config(run, single_source_stats),
        sort_keys=False,
        default_flow_style=False,
        width=1000,
    )
    return header(run) + body


def parse_single_source_stats(values: list[str]) -> dict[str, str]:
    stats: dict[str, str] = {}
    for value in values:
        source, sep, path = value.partition("=")
        if not sep or source not in SINGLE_SOURCES:
            raise ValueError(
                f"{value!r}: expected SOURCE=DIR with SOURCE in {SINGLE_SOURCES}"
            )
        path = path.rstrip("/")
        if not path.startswith("/climate-default/"):
            raise ValueError(f"{path} is not a /climate-default/ path")
        stats[source] = path
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--single-source-stats",
        nargs="+",
        default=[],
        metavar="SOURCE=DIR",
        help=(
            "Per-cell stats dir of each single source (c96amip, era5-1979), "
            "as recorded by submit_fm_v3_jobs.py --stats-jobs. Without it only "
            "the 5-source grid is written."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=RUN_CONFIGS_DIR,
        help="Write the configs here (default: run_configs/).",
    )
    args = parser.parse_args()
    try:
        stats = parse_single_source_stats(args.single_source_stats)
    except ValueError as error:
        parser.error(str(error))
    selected = multi_source_runs() + [
        run for run in single_source_runs() if run.source in stats
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for run in selected:
        path = args.out_dir / run.config_name
        path.write_text(render(run, stats))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
