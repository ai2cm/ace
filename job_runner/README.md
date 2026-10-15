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
  `<experiment_dir>`.
- An absolute path is taken as given; a relative path is resolved against the
  repository root, matching how `<experiment_dir>` is already interpreted.
- The resolved path must lie inside the repository root — `gantry` uploads the
  repository and runs the job from its root, so a config outside it is invisible
  to the job. The script exits non-zero naming the offending path.
- `--config-dir` is hyphenated, unlike the underscored `--atmos_stats` family.

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

**Script Initialization:**
- `init_script_environment()`: Initialize REPO_ROOT, GIT_BRANCH, and BEAKER_USERNAME variables
- `is_job_row()`: Fail for an input file's header, blank lines, and `#` comments
- `input_header()`: Print an input file's header from its `*_COLUMNS` array
- `parse_row()`: Set each `*_COLUMNS` variable from a pipe-delimited row
- `parse_dry_run_flag()`: Parse `--dry-run` flag from arguments
- `validate_config_once()`: Validate a config; under `--dry-run`, once per unique config and overrides
- `parse_config_dir_arg()`: Parse `--config-dir` argument into `CONFIG_DIR_OVERRIDE`
- `resolve_config_dir()`: Resolve `--config-dir` to a repo-root-relative path, rejecting one outside the repository root

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

### TAG Field (Optional)

All training, finetuning, and evaluation input files now include an optional `tag` field as the second column (immediately after `group`):
- **Position**: Field 2 (after `group`)
- **Purpose**:
  - Training/finetuning: Modifies the job name from `{GROUP}-train` to `{GROUP}-{TAG}-train`
  - Evaluation: Modifies the job name from `{GROUP}-evaluator_{CKPT}-{CONFIG}` to `{GROUP}-{TAG}-evaluator_{CKPT}-{CONFIG}`
- **Format**: Can be empty (leave blank between pipes) or contain an alphanumeric identifier
- **Propagation**: TAG is automatically written to `experiments.txt` by training scripts and read by evaluation scripts
