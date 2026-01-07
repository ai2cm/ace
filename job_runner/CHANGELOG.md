
# Change Log

https://github.com/ai2cm/full-model/compare/job_runner

## [0.0.1] - 2025-11-06

### Added

- job_runner

### Changed

- job_runner

### Fixed

- job_runner

## [x.x.x] - yyyy-mm-dd

### Added

- CHANGELOG.md
- [FEAT] Changelog TODO automation
- `make jr_change`

### Changed

### Fixed

- [ISSUE] `init_exper.sh` populates wrong dataset for `<COUPLED_ATMOS_ZARR>` in uncoupled atmosphere config.
- [ISSUE] `init_exper.sh` populates top-level `n_forward_steps` with inference value

## Releases

[unreleased]: https://github.com/ai2cm/full-model/compare/job_runner..ee9a4d2
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
- [ISSUE] `sed` usage not working on macOS
