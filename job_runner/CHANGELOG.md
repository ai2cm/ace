
# Change Log

https://github.com/ai2cm/full-model/compare/job_runner

## [Unreleased]

### Added

- CHANGELOG.md
- [FEAT] Changelog TODO automation
- `make jr_change`
- [FEAT] Add `--config-dir` arg to `{evaluate,inference}.sh` for shared eval
  configs

### Changed

- Remove `set_default_stats()`
- Allow `--atmos_stats` and `--ocean_stats` to be used separately.
- CLUSTER can be now be set for eval jobs
- Add "a100" and "h100" as additional CLUSTER options
- Allow use of `StandaloneComponentCheckpointsConfig` in eval
- Update config creation scripts for changes in PRs [#814](https://github.com/ai2cm/ace/pull/814) and [#862](https://github.com/ai2cm/ace/pull/862)

### Fixed

- [ISSUE] `init_exper.sh` populates wrong dataset for `<COUPLED_ATMOS_ZARR>` in uncoupled atmosphere config.
- [ISSUE] `init_exper.sh` populates top-level `n_forward_steps` with inference value
- [ISSUE] Use `ocean` as sub-directory of `--coupled_stats`.
- [ISSUE] Use `*=n` rather than `*=` in `create_coupled_finetune_config.sh`.
- [ISSUE] `create_coupled_train_config.sh` used the BSD-only `sed -i ''` form for
  the ocean stats rewrite, so every coupled submission from a Linux host aborted
  with `sed: can't read s/statsdata/ocean_stats/g: No such file or directory`.
  Now uses the portable `-i.bak` form already used for the atmosphere rewrite
  two lines above.
- [ISSUE] `create_input_txt_files()` wrote a 10-column `experiments.txt` header,
  but `evaluate.sh` and `inference.sh` read 14 fields — `cluster`,
  `ocean_results_dataset`, `atmos_results_dataset`, `shared_mem`. Coupled
  evaluation silently skipped the component checkpoint mounts and fell back to
  an empty cluster and shared memory. Header corrected in all three template
  branches, and in README.md.

## [0.0.1] - 2025-11-06

### Added

- job_runner

### Changed

- job_runner

### Fixed

- job_runner

## Releases

[unreleased]: https://github.com/ai2cm/full-model/compare/ee9a4d2..job_runner
[0.0.1] https://github.com/ai2cm/full-model/tree/ee9a4d2d3cc735027163fe56ae14c403c975484d

## TODO

- [REFACTOR] Nest config templates in `config_templates/{atmos,ocean,coupled,uncoupled}/` directories
- [REFACTOR] Move scripts to `lib/` directory
- [REFACTOR] Simplify README.md
- [FEAT] wandb project configurability
- [FEAT] Manage stats datasets in datasets.yaml
- [FEAT] Allow arbitrary config templates
- [FEAT] Interactive `init_exper.sh` + `make jr_init_exper`
- [FEAT] Slurm awareness
- [ISSUE] `init_exper.sh` populates top-level `n_coupled_steps` with inference value
- [ISSUE] Ignore header and empty lines in input .txt files when using `--dry-run`
- [ISSUE] Don't commit if gantry fails.
- [CHORE] Automate porting of experiment-branch-to-job_runner-branch updates
- [ISSUE] `make jr_changelog` doesn't properly handle insertion after multi-line
  items
