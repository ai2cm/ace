# Job Runner

This directory contains consolidated job submission scripts for training, fine-tuning, resuming, and evaluating climate models (both coupled and uncoupled).

## Overview

The job runner system consolidates common functionality from multiple training scripts into:
- **lib.sh**: Shared library with common functions for stats parsing, cluster configuration, and gantry job submission
- **lib_init_exper.sh**: Shared library for experiment directory initialization
- **Wrapper scripts**: Mode-specific scripts that handle job-specific logic (training, fine-tuning, resume, evaluation)
- **init_exper.sh**: Script to bootstrap new experiment directories from presets

## Architecture

```
job_runner/                       # Central job runner (this directory)
├── lib.sh                        # Shared library functions
├── lib_init_exper.sh             # Library for experiment initialization
├── init_exper.sh                 # Initialize new experiment directories
├── coupled_train.sh              # Coupled training wrapper
├── coupled_finetune.sh           # Coupled fine-tuning wrapper
├── uncoupled_train.sh            # Uncoupled training wrapper
├── uncoupled_finetune.sh         # Uncoupled fine-tuning wrapper
├── resume.sh                     # Resume training wrapper
├── evaluate.sh                   # Evaluation wrapper
├── config_templates/             # Config templates for initialization
│   ├── atmos.yaml
│   ├── ocean.yaml
│   ├── coupled.yaml
│   ├── finetune_coupled.yaml
│   ├── finetune_uncoupled.yaml
│   ├── eval_coupled.yaml
│   └── eval_uncoupled.yaml
├── expers/                       # Experiment presets
│   ├── cm4_piControl.yaml
│   └── cm4-train_1pctCO2-eval_piControl.yaml
└── datasets.yaml                 # Dataset references

configs/experiments/              # Experiment directories
└── <experiment_name>/
    ├── ocean/                    # Ocean model configs
    ├── atmos/                    # Atmosphere model configs
    └── coupled/                  # Coupled model configs
```

## Usage

### Running Job Scripts

All wrapper scripts are called directly from the `job_runner` directory:

```bash
cd job_runner

# Uncoupled training
./uncoupled_train.sh configs/experiments/my-exper/ocean config-v1

# Uncoupled fine-tuning
./uncoupled_finetune.sh configs/experiments/my-exper/ocean config-v1

# Coupled training
./coupled_train.sh configs/experiments/my-exper/coupled config-v1

# Coupled fine-tuning
./coupled_finetune.sh configs/experiments/my-exper/coupled config-v1

# Resume training (coupled or uncoupled)
./resume.sh configs/experiments/my-exper/ocean config-v1

# Evaluation (coupled or uncoupled)
./evaluate.sh configs/experiments/my-exper/ocean config-v1

# Evaluation / inference reading configs from a shared directory
./evaluate.sh configs/experiments/my-exper/ocean config-v1 --config-dir configs/baselines/cm4-piControl

# All scripts support stats overrides and dry-run:
./uncoupled_train.sh configs/experiments/my-exper/ocean config-v1 --atmos_stats path/to/stats --dry-run
./coupled_train.sh configs/experiments/my-exper/coupled config-v1 --coupled_stats path/to/stats --dry-run
```

### Initialize Experiment Presets

Use `init_exper.sh` to bootstrap a new experiment directory from the preset metadata in `job_runner/expers/`:

```bash
./job_runner/init_exper.sh \
    --exper cm4_piControl \
    --name ocean_baseline \
    --template ocean
```

- `--exper` selects the preset from `job_runner/expers/` directory, which specifies dataset groups, subsets, and inference windows.
- `--template` chooses one of `ocean`, `atmos`, or `coupled` in `job_runner/config_templates/`, or omit `--template` to initialize all templates.
- The script writes configs to `configs/experiments/<name>/<template>/` and pre-populates `train-config.yaml` (or `train-config-template.yaml` for coupled) plus the text-input stubs.

## Stats Dataset Options

All training scripts support three mutually exclusive options:

- `--atmos_stats <path>`: Override atmosphere stats dataset path
- `--ocean_stats <path>`: Override ocean stats dataset path
- `--coupled_stats <path>`: Use a coupled stats dataset with `coupled_atmosphere` and `uncoupled_ocean` subdirectories

If none are specified, defaults to:
- Atmosphere: `jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-atmosphere`
- Ocean: `jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-ocean`

## Shared Config Directory (`--config-dir`)

`evaluate.sh` and `inference.sh` accept an optional `--config-dir <path>` that
redirects where the config yaml is read from, so several experiment directories
can evaluate against one maintained set of `evaluator-config-*.yaml` /
`inference-config-*.yaml` files instead of each keeping a copy.

```bash
# default: configs are read from <experiment_dir>/<config_subdirectory>/
./job_runner/evaluate.sh configs/experiments/my-exper/coupled v1 --dry-run

# shared: configs are read from the given directory instead
./job_runner/evaluate.sh configs/experiments/my-exper/coupled v1 \
    --config-dir configs/baselines/cm4-piControl --dry-run
```

- The config filename is unchanged: `<config-dir>/evaluator-config-<tag>.yaml`,
  with `<tag>` still the `experiments.txt` row's `status` field minus its `run_`
  prefix (`run_inf_` for `inference.sh`).
- `experiments.txt` is **always** read from
  `<experiment_dir>/<config_subdirectory>/`. The shared directory supplies
  configs, not the job list.
- Module selection (`fme.coupled.*` vs `fme.ace.*`) stays keyed off
  `<experiment_dir>`: `fme.coupled.*` when one of its path components starts
  with `coupled` (`coupled`, `coupled_ft`), so `uncoupled/ocean` gets `fme.ace.*`.
- An absolute path is taken as given; a relative path is resolved against the
  repository root, matching how `<experiment_dir>` is already interpreted.
- The resolved path must lie inside the repository root — `gantry` uploads the
  repository and runs the job from its root, so a config outside it is invisible
  to the job. The script exits non-zero naming the offending path.
- `--config-dir` is hyphenated, unlike the underscored `--atmos_stats` family.

## Queued Evaluations (`--eval-config-dir`)

`uncoupled_train.sh`, `coupled_train.sh`, `uncoupled_finetune.sh` and
`coupled_finetune.sh` accept an optional `--eval-config-dir <path>`. For each
submitted run, they append one `wait_<tag>` row per
`evaluator-config-<tag>.yaml` in that directory to `experiments.txt`, and no
`training` row. Without the flag, they append one `training` row as before.

```bash
./job_runner/uncoupled_train.sh configs/experiments/my-exper/ocean config-v1 \
    --ocean_stats path/to/stats --eval-config-dir configs/experiments/cm4_eval/ocean --dry-run
```

```
group|tag|experiment_id|status|checkpoint|priority|min_runtime|...|n_gpus|segments
G|rs0|<training experiment ID>|wait_<tag>|best_inference_ckpt|high|--min-runtime 8h|...|1|
```

- `<tag>` is the config filename minus `evaluator-config-` and `.yaml`, so
  flipping `wait_<tag>` to `run_<tag>` makes `evaluate.sh --config-dir <path>`
  launch that config against the training's checkpoint.
- `experiment_id` is the training's beaker experiment ID; `evaluate.sh`
  resolves its results dataset from it.
- Row settings are the `EVAL_WAIT_*` variables in `lib.sh`, except where the flags below override them.
- `--eval-checkpoints <ckpt>[,<ckpt>...]` (needs `--eval-config-dir`) writes
  one row per config and checkpoint, the checkpoint in the `checkpoint` column,
  checkpoints varying fastest; without it each row uses `EVAL_WAIT_CHECKPOINT`.
  Names are `[A-Za-z0-9_.]`: a `-` could make two job names identical.
  `evaluate.sh` puts the checkpoint in the job group
  (`<group>-eval_<ckpt>-<tag>`), so job and wandb names differ per checkpoint:

  ```
  G|rs0|<training experiment ID>|wait_1pct|ema_ckpt_0001|high|--min-runtime 8h|...|1
  G|rs0|<training experiment ID>|wait_1pct|ema_ckpt_0002|high|--min-runtime 8h|...|1
  ```
- `--eval-priority <priority>` and `--eval-cluster <cluster>` (each needs
  `--eval-config-dir`) set the `priority` and `cluster` columns of every row;
  `evaluate.sh` passes them to gantry. Priority is one of `EVAL_PRIORITIES`
  (`low normal high urgent`); cluster is one of `EVAL_CLUSTER_ALIASES` (the
  `build_cluster_args` aliases) or a `node_caps.txt` cluster, as `<name>` or
  `ai2/<name>`. Anything else stops the script before any submission. Without
  them, rows keep `EVAL_WAIT_PRIORITY` and `EVAL_WAIT_CLUSTER` (blank). With
  `--eval-checkpoints ema_ckpt_0001 --eval-priority urgent --eval-cluster titan`:

  ```
  G|rs0|<training experiment ID>|wait_1pct|ema_ckpt_0001|urgent|--min-runtime 8h||||titan||||1|
  ```
- The directory resolves as `--config-dir` does (absolute, or relative to the
  repository root, and inside it). A directory with no
  `evaluator-config-*.yaml` stops the script before any submission.
- `--dry-run` prints the rows each run would get.
- Nothing flips `wait_` rows; that is the caller's job (by hand or a watcher).

## Atmosphere of a Coupled Checkpoint (`--lift-coupled-atmosphere`)

`evaluate.sh --lift-coupled-atmosphere` runs an atmosphere-only evaluator
config on rows whose checkpoint is a coupled one. Each job first runs
`bin/extract_component_from_coupled_ckpt.py /ckpt.tar /tmp/lifted_atmosphere_ckpt.tar --component atmosphere`,
then `fme.ace.evaluator` with `checkpoint_path=/tmp/lifted_atmosphere_ckpt.tar`
appended to the overrides. No beaker dataset is created for the lifted file.

```bash
./job_runner/evaluate.sh configs/experiments/my-exper/atmos_from_coupled . \
    --config-dir configs/experiments/cm4_eval/atmos --lift-coupled-atmosphere --dry-run
```

- Any `<experiment_dir>`, coupled included: the flag selects
  `fme.ace.validate_config` and `fme.ace.evaluator` whatever the path.
- Applies to every `run_<tag>` row of the file. In a coupled
  `experiments.txt`, a pending coupled-eval row whose tag also names an
  atmosphere config in `--config-dir` (for example `run_1pct-ic0001-130yr`
  when both `eval/atmos/` and `eval/coupled/` hold
  `evaluator-config-1pct-ic0001-130yr.yaml`) launches as a lifted-atmosphere
  eval: a valid job on the wrong config. Before a lift launch, flip or remove
  such rows, or keep coupled-checkpoint atmosphere rows in their own
  `experiments.txt`. Rows whose tag is absent from `--config-dir` print
  `Config file not found ... Skipping`.
- Any checkpoint file works. `ckpt.tar` holds raw weights plus EMA weights;
  the lifted file keeps the atmosphere's EMA weights and the evaluator uses
  them (`use_ema_if_available`, default true). `ema_ckpt_*.tar` and
  `best_*ckpt.tar` already hold EMA weights.
- The job runs the launching branch's fme. A checkpoint whose config that fme
  cannot load (written by a newer fme) fails the job before the evaluator
  starts; the log names the git sha it was trained at. Launch such evals from
  a branch with that fme that has merged `job_runner`.

### Lifting by hand

`bin/extract_component_from_coupled_ckpt.py` lifts either component:

```bash
python job_runner/bin/extract_component_from_coupled_ckpt.py \
    <coupled ckpt.tar> <out.tar> --component {atmosphere,ocean}
```

The output loads with `fme.ace.evaluator` (`checkpoint_path: <out.tar>`).

## Dry-Run Mode

All wrapper scripts support a `--dry-run` flag that allows you to preview actions without launching jobs or committing changes:

```bash
# Preview training jobs without execution
./job_runner/uncoupled_train.sh configs/experiments/my-exper/ocean . --dry-run

# Preview evaluation jobs
./job_runner/evaluate.sh configs/experiments/my-exper/coupled . --dry-run
```

In dry-run mode:
- No gantry jobs are launched
- No git commits or pushes are made
- No files are modified
- The environment header names the directory configs are read from
- Detailed output shows what would be executed:
  - First job: Full detailed configuration
  - Subsequent jobs: Condensed summary
  - Final summary: Total jobs, processed count, and skipped count

This is useful for:
- Validating input file formats
- Checking job configurations before submission
- Testing changes to wrapper scripts
- Reviewing multiple job submissions at once

## Shared Library Functions

### `lib.sh` Functions

**Stats and Dataset Management:**
- `parse_stats_args()`: Parse `--atmos_stats`, `--ocean_stats`, `--coupled_stats` arguments
- `validate_stats_args()`: Ensure mutual exclusivity of stats options
- `build_stats_dataset_args()`: Construct `STATS_DATASET_ARGS` array for dataset mounting
- `print_stats_config()`: Display stats configuration for debugging

**Cluster and Job Configuration:**
- `build_cluster_args()`: Construct `CLUSTER_ARGS` array for gantry
- `build_inference_segments_args()`: Set `SEGMENTS_ARGS` to `--segments N` from the `segments` column (blank: none); exit on a non-positive-integer value
- `default_shared_mem()`: Fill a blank `shared_mem` column from `node_caps.txt`:
  `SHARED_MEM_FRACTION` (default 0.8) x `n_gpus/gpus_per_node` x node memory
  limit, min over the alias's clusters; unknown clusters fall back to
  `n_gpus` x 64GiB
- `get_beaker_username()`: Get Beaker username from account info
- `build_job_name()`: Build job name from group, tag, and suffix

**Beaker and WandB Integration:**
- `get_beaker_dataset_from_experiment()`: Fetch dataset ID from experiment ID
- `get_experiment_from_wandb()`: Convert wandb project/run to beaker experiment ID

**Job Execution:**
- `run_gantry_training_job()`: Execute gantry training job with common parameters
- `run_gantry_training_job_with_dry_run()`: Wrapper for gantry job that respects dry-run mode

**Git Operations:**
- `git_commit_and_push()`: Commit and push files to git
- `git_commit_and_push_with_dry_run()`: Wrapper for git operations that respects dry-run mode
- `append_to_experiments_file()`: Append experiment result to experiments.txt and commit
- `append_to_experiments_file_with_dry_run()`: Wrapper for appending to experiments.txt that respects dry-run mode
- `append_rows_to_experiments_file()`: Append a block of rows to experiments.txt and commit
- `experiments_row()`: Print an experiments.txt row from `<header name>=<value>` pairs
- `eval_wait_rows()`: Print a training's `wait_<tag>` rows, one per `EVAL_CONFIG_TAGS` entry
- `record_training_with_dry_run()`: Append a submitted training's `training` row, or its `wait_<tag>` rows under `--eval-config-dir`

**Script Initialization:**
- `init_script_environment()`: Initialize REPO_ROOT, GIT_BRANCH, and BEAKER_USERNAME variables
- `is_job_row()`: Fail for an input file's header, blank lines, and `#` comments
- `input_header()`: Print an input file's header from its `*_COLUMNS` array
- `parse_row()`: Set each `*_COLUMNS` variable from a pipe-delimited row
- `parse_dry_run_flag()`: Parse `--dry-run` flag from arguments
- `validate_config_once()`: Validate a config; under `--dry-run`, once per unique config and overrides
- `parse_config_dir_arg()`: Parse `--config-dir` argument into `CONFIG_DIR_OVERRIDE`
- `resolve_config_dir()`: Resolve `--config-dir` to a repo-root-relative path, rejecting one outside the repository root
- `parse_eval_config_dir_arg()`: Parse `--eval-config-dir`, `--eval-checkpoints`, `--eval-priority` and `--eval-cluster` into `EVAL_*_OVERRIDE` and `EVAL_*_GIVEN`
- `resolve_eval_config_dir()`: Resolve `--eval-config-dir` into `EVAL_CONFIG_DIR`, its `EVAL_CONFIG_TAGS`, `EVAL_CHECKPOINTS`, `EVAL_PRIORITY` and `EVAL_CLUSTER`
- `resolve_eval_checkpoints()`: Split and check `--eval-checkpoints` into `EVAL_CHECKPOINTS`
- `resolve_eval_priority_and_cluster()`: Check `--eval-priority` / `--eval-cluster` into `EVAL_PRIORITY` / `EVAL_CLUSTER`
- `node_caps_clusters()`: Print the cluster names in `node_caps.txt`
- `is_one_of()`: Succeed when a value equals one of the listed values

**Output and Logging:**
- `print_dry_run_header()`: Print dry-run mode header
- `print_detailed_job_info()`: Print detailed job info for first job
- `print_condensed_job_info()`: Print condensed job summary for subsequent jobs
- `print_dry_run_summary()`: Print dry-run summary at the end

## Input File Formats

Each job type reads one pipe-delimited text file. Its columns are the
`*_COLUMNS` array in `lib.sh`, the single source for both the header and how
rows are parsed (`parse_row`); running a launcher with no arguments prints its
header.

| Job type | File | Columns |
| --- | --- | --- |
| Uncoupled training | `training.txt` | `TRAINING_COLUMNS` |
| Coupled training | `pretraining.txt` | `PRETRAINING_COLUMNS` |
| Fine-tuning (coupled or uncoupled) | `finetuning.txt` | `FINETUNING_COLUMNS` |
| Resume | `resuming.txt` | `RESUMING_COLUMNS` |
| Evaluate / Inference | `experiments.txt` | `EXPERIMENTS_COLUMNS` |

`experiments.txt` is always read from `<experiment_dir>/<config_subdirectory>/`,
never from `--config-dir`. Training scripts create it with its header if it is
missing. In `resuming.txt`, the optional `config` column names a repo config in
the config subdirectory, used in place of the resumed run's dumped config.

### `min_runtime` Field (Optional, Training Inputs)

All training-related input files (`pretraining.txt`, `training.txt`, `finetuning.txt`,
`resuming.txt`) include an optional `min_runtime` field. It is
passed directly to gantry as `--min-runtime <value>` and controls how long a job
is protected from preemption before it can be interrupted.

- **Format**: A gantry-style duration string (e.g. `0`, `30m`, `1h`, `8h`).
- **Default**: `DEFAULT_MIN_RUNTIME` in `lib.sh` (`8h`), for training, evaluation,
  and inference rows alike.
- **Auto-resume**: Training jobs always have auto-resume enabled (server default);
  `--no-auto-resume` is never emitted on the training path.
- **Example values**: `1h` to protect a checkpoint cycle, `8h` for a fully
  protected run. `0` is backfill-only.

### `n_gpus` Field (Optional, `experiments.txt`)

The `n_gpus` column sets how many GPUs `evaluate.sh` and
`inference.sh` request (`gantry run --gpus`) and how the fme module starts.

| `n_gpus` | launch |
| --- | --- |
| blank (or column absent) | as `1` |
| `1` | `python -I -m <module>` |
| `N` > 1 | `torchrun --nproc_per_node N -m <module>`, one node |
| anything else | the script exits before submitting |

- `n_initial_conditions` must be divisible by `N`; fme raises at job start,
  not at `validate_config`.
- `inference.sh` exits for `N` > 1 when it would run `fme.coupled.inference`,
  which does not shard initial conditions across GPUs (ai2cm/ace#1635).
  Coupled `evaluate.sh` rows are allowed.
- A blank `shared_mem` scales with `N` (`default_shared_mem`).

### `segments` Field (Optional, `experiments.txt`)

The last `EXPERIMENTS_COLUMNS` column runs an `inference.sh` row as a
segmented, preemption-resumable inference (`--segments`, ai2cm/ace#1368):
`N` segments of `n_coupled_steps` (coupled) or `n_forward_steps` (atmosphere)
each, written to per-segment subdirectories of `experiment_dir`. A segment
whose restart files exist is skipped on rerun.

| `segments` | `inference.sh` |
| --- | --- |
| blank (or column absent) | no `--segments`; gantry `--no-auto-resume` |
| `N` (positive integer) | `<module> <config> --segments N --override ...`; gantry auto-resume kept |
| anything else | the script exits before submitting |

- `--segments` reaches only the inference command, never `validate_config`.
- With auto-resume, beaker restarts a preempted job, which finds the finished
  segments under `experiment_dir` and resumes at the first unfinished one.
  This needs `experiment_dir` to survive the restart: `/results` (where
  training's auto-resume reloads its checkpoints from) or a `gs://` path. Any
  other local path starts over at segment 0.
- `n_ensemble_per_ic` must be 1; fme raises at job start, not at
  `validate_config`.
- `evaluate.sh` exits on a non-blank `segments`: the evaluators have no
  `--segments` flag.

### TAG Field (Optional)

All training, finetuning, and evaluation input files now include an optional `tag` field as the second column (immediately after `group`):
- **Position**: Field 2 (after `group`)
- **Purpose**:
  - Training/finetuning: Modifies the job name from `{GROUP}-train` to `{GROUP}-{TAG}-train`
  - Evaluation: Modifies the job name from `{GROUP}-evaluator_{CKPT}-{CONFIG}` to `{GROUP}-{TAG}-evaluator_{CKPT}-{CONFIG}`
- **Format**: Can be empty (leave blank between pipes) or contain an alphanumeric identifier
- **Propagation**: TAG is automatically written to `experiments.txt` by training scripts and read by evaluation scripts
