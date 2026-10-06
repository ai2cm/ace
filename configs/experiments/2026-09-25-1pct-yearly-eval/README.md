# 1pctCO2 full-period evaluation (monthly output)

Shared evaluator config for running any 4deg ocean checkpoint over the full
1pctCO2 dataset (139 years, 0002-01-01 to 0141-01-01) and writing monthly
means plus yearly-mean corrector deltas. Uses the job runner's
`--config-dir` mode: the config lives here, the job list (`experiments.txt`)
lives with each training experiment and just names the checkpoint.

| config | status tag | for checkpoints trained on |
| --- | --- | --- |
| `evaluator-config-1pct_yearly_4deg.yaml` | `run_1pct_yearly_4deg` | 4deg zarrs (`2026-07-22-cm4-*-4deg-coupled-ocean`, `2026-07-15-om4-*-4deg-ocean-5daily`) |
| `evaluator-config-1pct_yearly_4deg_icevol_ssf.yaml` | `run_1pct_yearly_4deg_icevol_ssf` | `2026-09-30-resid-thetao-no-sal-baseline` only; same as above plus a `stepper_override` that repeats that run's corrector (surface energy flux `prescribed`) and adds the sea-surface-fraction-weighted ice volume salt correction |
| `evaluator-config-1pct_yearly_4deg_icevol_ssf_open.yaml` | `run_1pct_yearly_4deg_icevol_ssf_open` | the no-salt runs trained with surface energy flux `prescribed_open_ocean` (`2026-09-28-test-no-sal-correct`, `2026-09-29-test-resid-no-sal-correct`, `2026-10-01-direct-ssh-no-sal-baseline`); the same salt correction with that corrector repeated |
| `evaluator-config-1pct_yearly_4deg_ssh_open.yaml` | `run_1pct_yearly_4deg_ssh_open` | `2026-10-01-direct-ssh-no-sal-baseline` only (needs `SSH`, not `zos`); the `_icevol_ssf_open` corrector with the salt budget from the change of the sea surface height |
| `evaluator-config-1pct_yearly_4deg_ssh_brine_open.yaml` | `run_1pct_yearly_4deg_ssh_brine_open` | `2026-10-01-direct-ssh-no-sal-baseline` only; as `_ssh_open` plus the predicted `sfdsi` in the budget |
| `evaluator-config-1pct_yearly_4deg_wfo_regimes_open.yaml` | `run_1pct_yearly_4deg_wfo_regimes_open` | the `prescribed_open_ocean` no-salt runs; the `_icevol_ssf_open` corrector with a water flux salt budget: P − E over open water, the sea ice volume change under ice, the predicted `wfo` at ice-free coasts, plus the predicted `sfdsi` |
| `evaluator-config-1pct_yearly_4deg_wfo_pe_open.yaml` | `run_1pct_yearly_4deg_wfo_pe_open` | the `prescribed_open_ocean` no-salt runs; as `_wfo_regimes_open` but with the predicted `wfo` under ice too |
| `evaluator-config-1pct_yearly_4deg_wfo_fullice_open.yaml` | `run_1pct_yearly_4deg_wfo_fullice_open` | the `prescribed_open_ocean` no-salt runs; as `_wfo_regimes_open` but with the sea ice volume term only under full ice cover (fraction at least 0.99 at both steps), and the predicted `wfo` under partial ice |
| `evaluator-config-1pct_yearly_4deg_wfo_frc_open.yaml` | `run_1pct_yearly_4deg_wfo_frc_open` | the `prescribed_open_ocean` no-salt runs; water flux salt budget with the target `wfo` and `sfdsi` from the data (`fluxes_from_forcing`) everywhere |
| `evaluator-config-1pct_yearly_4deg_wfo_pe_frc_open.yaml` | `run_1pct_yearly_4deg_wfo_pe_frc_open` | as `_wfo_pe_open`, with the target `wfo` and `sfdsi` in place of the predicted ones |
| `evaluator-config-1pct_yearly_4deg_wfo_fullice_frc_open.yaml` | `run_1pct_yearly_4deg_wfo_fullice_frc_open` | as `_wfo_fullice_open`, with the target `wfo` and `sfdsi` in place of the predicted ones |
| `evaluator-config-1pct_yearly_4deg_ssh_brine_comp_open.yaml` | `run_1pct_yearly_4deg_ssh_brine_comp_open` | `2026-10-01-direct-ssh-no-sal-baseline` only; as `_ssh_brine_open`, with the sea ice salt flux computed from the predicted sea ice volume (S_ice 3.0 psu) wherever there is ice, in place of the predicted `sfdsi`; nothing from the forcing data |
| `evaluator-config-1pct_yearly_4deg_ssh_brine_frc_open.yaml` | `run_1pct_yearly_4deg_ssh_brine_frc_open` | `2026-10-01-direct-ssh-no-sal-baseline` only; as `_ssh_brine_open`, with the target `sfdsi` in place of the predicted one |

A 1deg checkpoint needs a twin config with the 1deg stores
(`2026-07-15-om4-1pctco2-1deg-coupled-ocean`,
`2026-07-15-om4-1pctco2-1deg-ocean-5daily`); the time axis is the same, so
only the `file_pattern` entries change.

## Running

From the repo root, one call per experiment directory (add `--dry-run` to
preview). `experiments.txt` is always read from the experiment directory;
only the config yaml is read from here.

```bash
bash job_runner/evaluate.sh configs/experiments/2026-09-28-test-no-sal-correct . \
    --config-dir configs/experiments/2026-09-25-1pct-yearly-eval
```

The salinity-corrector runs and the `_totalsalt64` variant of this config,
which adds the salt correction through `stepper_override.corrector`, are on
`exp/2026-09-30-salt-corrector-pr1533`.

## Adding a checkpoint

Append a row to the `experiments.txt` of the training experiment (the
training runner writes `training` rows there; those are skipped by
`evaluate.sh`), with the status set to the tag of the matching config:

```
<group>|<tag>|<beaker experiment id>|run_1pct_yearly_4deg|best_inference_ckpt|normal|--min-runtime 8h
```

`evaluate.sh` looks up the results dataset from the experiment id. If that
lookup comes back empty ("dataset id is required"), put the results dataset
id in column 9 instead:

```
<group>|<tag>|<exper id>|run_1pct_yearly_4deg|best_inference_ckpt|normal|--min-runtime 8h||<results dataset id>
```

Column 8 (`override_args`) takes `--override` dotlist entries applied on top
of the config.
