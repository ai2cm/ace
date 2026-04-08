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
- `parse_dry_run_flag()`: Parse `--dry-run` flag from arguments

**Output and Logging:**
- `print_dry_run_header()`: Print dry-run mode header
- `print_detailed_job_info()`: Print detailed job info for first job
- `print_condensed_job_info()`: Print condensed job summary for subsequent jobs
- `print_dry_run_summary()`: Print dry-run summary at the end

## Input File Formats

Each job type reads from a pipe-delimited text file:

- **Coupled training**: `pretraining.txt` (16 fields)
- **Uncoupled training**: `training.txt` (10 fields)
- **Coupled fine-tuning**: `finetuning.txt` (14 fields)
- **Uncoupled fine-tuning**: `finetuning.txt` (14 fields)
- **Resume**: `resuming.txt` (13 fields)
- **Evaluate**: `experiments.txt` (10 fields)

### TAG Field (Optional)

All training, finetuning, and evaluation input files now include an optional `tag` field as the second column (immediately after `group`):
- **Position**: Field 2 (after `group`)
- **Purpose**:
  - Training/finetuning: Modifies the job name from `{GROUP}-train` to `{GROUP}-{TAG}-train`
  - Evaluation: Modifies the job name from `{GROUP}-evaluator_{CKPT}-{CONFIG}` to `{GROUP}-{TAG}-evaluator_{CKPT}-{CONFIG}`
- **Format**: Can be empty (leave blank between pipes) or contain an alphanumeric identifier
- **Propagation**: TAG is automatically written to `experiments.txt` by training scripts and read by evaluation scripts
- **Example headers**:
  - Uncoupled training: `group|tag|skip_or_train|priority|...`
  - Coupled training: `group|tag|ocean_project|ocean_wandb_id|...`
  - Uncoupled finetuning: `group|tag|wandb_project|wandb_id|...`
  - Coupled finetuning: `group|tag|wandb_project|wandb_id|...`
  - Experiments: `group|tag|exper_id|status|ckpt_type|priority|...`
