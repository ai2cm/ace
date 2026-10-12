# AGENTS.md — `job_runner`

Traps and standing defaults for submitting jobs with these scripts. Input-file
columns are the `*_COLUMNS` arrays in `lib.sh`; a launcher run with no
arguments prints its header. `README.md` lists the `lib.sh` functions.

## Committing

A commit that modifies files in `job_runner/` touches nothing outside
`job_runner/`, and its message begins with `job_runner: `.

## Before any submission

**Check the branch.** The scripts commit the generated config and an
`experiments.txt` row, then push to the branch `init_script_environment` reads
from HEAD (`git_commit_and_push`). Check out the experiment directory's branch
first.

**Dry-run first.** `--dry-run` launches, commits, and pushes nothing (the
`*_with_dry_run` wrappers in `lib.sh`). Confirm the parsed fields, job name,
and config path before submitting.

**`min_runtime` defaults to `8h`** (`DEFAULT_MIN_RUNTIME` in `lib.sh`) when left
blank. `0` is backfill-only.

## Resuming a completed or stopped run

`resume.sh <experiment_dir> <config_subdir>` reads `resuming.txt`. Typical use
is extending a finished run to a larger `max_epochs`.

### Which config a resume runs

**`config` blank (default):** the job runs `/existing-results/config.yaml`, the
resolved config the previous run dumped. **Editing the repo's
`train-config.yaml` has no effect**; every change goes through `override`. The
original submission's overrides are already baked in, so do not repeat them.

**`config` set:** a file name in `<experiment_dir>/<config_subdir>/` replaces
the dumped config. Use it when a change touches a list (next section). The job
runs the committed, pushed copy, not the working tree. It carries none of the
original overrides, so **do** repeat them (`seed`, per-row `block_size`, ...)
in `override`.

Either way, `resume.sh` prepends `resume_results.existing_dir=/existing-results`
to `override`; never write it yourself. The dataset must contain `config.yaml`,
`wandb_run_id`, and `training_checkpoints/ckpt.tar`.

### Changing one entry of a list means replacing the whole list

`override` is parsed by `OmegaConf.from_dotlist` (`update_dict_with_dotlist` in
`fme/core/config.py`), which has no list-index syntax.
`validation.0.evaluate_all_steps=false` builds a dict keyed `"0"`, and the
merge raises

    ConfigTypeError: Cannot merge DictConfig with ListConfig

So either point `config` at a fixed repo config (preferred), or inline the
**entire** list as one override value. `run_gantry_training_job` re-tokenizes
`OVERRIDE_ARGS` through `eval`, so a single-quoted flow-style YAML value stays
one argv element:

    'validation=[{"name": "val_a", "evaluate_all_steps": false, ...}, {...}]'

- **Generate it from `/existing-results/config.yaml`; never transcribe it.**
  Change only the keys you mean to, and assert that reverting them reproduces
  the original block.
- **Double quotes inside, single quotes outside.** A single quote inside ends
  the bash word, and `eval` splits the value.

**`resume.sh` never runs `validate_config`**, unlike the other six scripts, and
`--dry-run` never resolves the config, so a bad override fails on the cluster.
Validate by hand first:

    python -m fme.ace.validate_config <dumped config> --config_type train \
      --override resume_results.existing_dir=/existing-results <your overrides>

### Resume defaults

**Leave `results_dataset` blank.** `resume.sh` then resolves wandb run →
`BEAKER_EXPERIMENT_ID` → that experiment's `jobs[-1].result`
(`get_experiment_from_wandb`, `get_beaker_dataset_from_experiment`). With
`resume_wandb=true` that key is overwritten on every resume, so the lookup
tracks the newest checkpoint. A pinned dataset id silently resumes from a
stale epoch on the next extension.

**Keep `tag` unchanged.** The resume continues the same seed and wandb run.
The duplicate Beaker name `{group}-{tag}-train` (`build_job_name`) is fine:
Beaker appends a 4-char suffix on collision.

**Set `resume_results.resume_wandb=true`** to continue the original wandb run
and keep the `results_dataset` lookup current. It defaults to `False`
(`ResumeResultsConfig` in `fme/core/cli.py`).

### Check the LR scheduler before raising `max_epochs`

A `CosineAnnealingLR` without an explicit `T_max` takes `T_max = max_epochs`
(`SchedulerConfig.build` in `fme/core/scheduler.py`). Raising `max_epochs`
then stretches the cosine under a restored scheduler state, so the resumed
epochs follow a different LR curve. `SequentialSchedulerConfig` has the same
problem via `milestones`.

Read `optimization.scheduler` in `/existing-results/config.yaml`. A
constant-`lr` optimizer with no `scheduler` key extends cleanly; anything else
needs a deliberate LR decision, recorded alongside the run.

## Reading results

Only the **final** job describes an experiment's outcome:
`jq '.[0].jobs[-1].result.beaker'` and `jobs[-1].status.exitCode`. A high job
count is normal. Preemption, an unhealthy node, and a crash all show
`exitCode: 1`; tell them apart with `jobs[].status.canceledFor`. Gantry
environment-validation failures exit in under a second, before any Python
runs.
