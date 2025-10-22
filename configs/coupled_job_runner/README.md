# Coupled Job Runner

This directory contains consolidated job submission scripts for training, fine-tuning, resuming, and evaluating climate models (both coupled and uncoupled).

## Overview

The job runner system consolidates common functionality from multiple training scripts into:
- **lib.sh**: Shared library with common functions for stats parsing, cluster configuration, and gantry job submission
- **Wrapper scripts**: Mode-specific scripts that handle job-specific logic
- **Shim scripts**: Thin compatibility layers in experiment directories that maintain backward compatibility

## Architecture

```
configs/
├── coupled_job_runner/           # Central job runner (this directory)
│   ├── lib.sh                    # Shared library functions
│   ├── coupled_train.sh          # Coupled training wrapper
│   ├── coupled_finetune.sh       # Coupled fine-tuning wrapper
│   ├── uncoupled_train.sh        # Uncoupled training wrapper
│   ├── uncoupled_finetune.sh     # Uncoupled fine-tuning wrapper
│   ├── resume.sh                 # Resume training wrapper
│   └── evaluate.sh               # Evaluation wrapper
└── experiments/
    └── 2025-08-08-jamesd/       # Example experiment directory
        ├── coupled/
        │   ├── train.sh          # Shim → coupled_train.sh
        │   └── finetune.sh       # Shim → coupled_finetune.sh
        ├── uncoupled/
        │   ├── train.sh          # Shim → uncoupled_train.sh
        │   └── finetune.sh       # Shim → uncoupled_finetune.sh
        ├── resume.sh             # Shim → resume.sh
        └── evaluate.sh           # Shim → evaluate.sh
```

## Usage

### From Experiment Directory (Recommended)

Scripts can be called from within the experiment directory as before:

```bash
cd configs/experiments/2025-08-08-jamesd

# Coupled training
./coupled/train.sh v2025-06-03-fto --coupled_stats dataset/path

# Uncoupled training
./uncoupled/train.sh some-config --atmos_stats dataset/path

# Fine-tuning
./coupled/finetune.sh v2025-06-03-fto
./uncoupled/finetune.sh some-config

# Resume training
./resume.sh coupled/v2025-06-03-fto --coupled_stats dataset/path

# Evaluation
./evaluate.sh coupled/v2025-06-03-fto
```

### Direct Wrapper Usage

Wrappers can also be called directly:

```bash
cd configs/coupled_job_runner

# Coupled training
./coupled_train.sh experiments/2025-08-08-jamesd/coupled v2025-06-03-fto

# Resume training
./resume.sh experiments/2025-08-08-jamesd/coupled v2025-06-03-fto
```

## Stats Dataset Options

All training scripts support three mutually exclusive options:

- `--atmos_stats <path>`: Override atmosphere stats dataset path
- `--ocean_stats <path>`: Override ocean stats dataset path
- `--coupled_stats <path>`: Use a coupled stats dataset with `coupled_atmosphere` and `uncoupled_ocean` subdirectories

If none are specified, defaults to:
- Atmosphere: `jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-atmosphere`
- Ocean: `jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-ocean`

## Shared Library Functions

### `lib.sh` Functions

- `parse_stats_args()`: Parse `--atmos_stats`, `--ocean_stats`, `--coupled_stats` arguments
- `validate_stats_args()`: Ensure mutual exclusivity of stats options
- `set_default_stats()`: Set default stat dataset paths
- `build_cluster_args()`: Construct `CLUSTER_ARGS` array for gantry
- `build_stats_dataset_args()`: Construct `STATS_DATASET_ARGS` array for dataset mounting
- `get_beaker_dataset_from_experiment()`: Fetch dataset ID from experiment ID
- `get_experiment_from_wandb()`: Convert wandb project/run to beaker experiment ID
- `git_commit_and_push()`: Commit and push files to git
- `run_gantry_training_job()`: Execute gantry training job with common parameters
- `print_stats_config()`: Display stats configuration for debugging

## Input File Formats

Each job type reads from a pipe-delimited text file:

- **Coupled training**: `pretraining.txt` (15 fields)
- **Uncoupled training**: `training.txt` (9 fields)
- **Coupled fine-tuning**: `finetuning.txt` (13 fields)
- **Uncoupled fine-tuning**: `finetuning.txt` (13 fields)
- **Resume**: `resuming.txt` (12 fields)
- **Evaluate**: `experiments.txt` (9 fields)

## Extending to Other Experiment Directories

To use this system in a new experiment directory:

1. Create shim scripts in your experiment directory (copy from `2025-08-08-jamesd/`)
2. Update `SCRIPT_DIR` path resolution in shims if needed
3. Ensure your input files follow the expected formats
4. Create any necessary config creation scripts

## Benefits

- **90% code reduction** in wrapper scripts (15 lines vs. 160-210 lines)
- **Single source of truth** for common operations
- **Easy maintenance** - bug fixes benefit all scripts
- **Backward compatible** - existing workflows unchanged
- **Extensible** - easy to add new modes or features

## Future Improvements

- Add validation for input file formats
- Support for additional cluster configurations
- Enhanced error handling and logging
- Configuration file for defaults
- Integration tests for all job types
