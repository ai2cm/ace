# AGENTS.md — `job_runner`

Agent guidance for submitting jobs with these scripts. `README.md` is the
reference for the pipe-delimited input-file formats and the `lib.sh` function
list; this file covers the traps those tables do not, and the owner's standing
defaults.

## Before any submission

**Check the branch.** `init_script_environment` reads `GIT_BRANCH` from HEAD
(`lib.sh:316-327`), and the scripts commit the generated config plus an
`experiments.txt` row and then `git push origin "$GIT_BRANCH"` (`lib.sh:162`).
Submitting from the wrong branch publishes those rows to it. When an experiment
directory belongs to a specific branch, check that branch out first.

**Never create a git worktree of `ace-exper`.** Owner's standing call
(2026-08-02): worktrees are worth their management overhead in `ace/` and for
special cases, not here. Branches are checked out in the main clone. The generic
worktree convention in the workspace's `docs/agents/branch-records.md` does not
apply to this repo.

**Dry-run first.** `--dry-run` launches nothing, commits nothing, pushes nothing
(`lib.sh:401-422`). Confirm the parsed fields, the job name, and the config path
before submitting for real.

**Expect `Will Skip` ≥ 1.** Every line of the input file is parsed, including the
header and the conventional blank line after it; rows whose `skip_or_train` field
is not `train` are skipped. A one-row file reports `Will Process: 1, Will Skip: 2`.

## Resuming a completed or stopped run

`resume.sh <experiment_dir> <config_subdir>` reads `resuming.txt` (16 fields):

```
group|tag|wandb_project|wandb_id|skip_or_train|priority|cluster|n_gpus|shared_mem|retries|workspace|override|results_dataset|results_dataset_ocean|results_dataset_atmos|min_runtime
```

Typical use is extending a run that finished all its epochs — train for a larger
`max_epochs` than originally configured.

### The config does not come from the repo

`resume.sh` sets the config to `/existing-results/config.yaml` (`resume.sh:157`),
mounted from the previous run's results dataset. **Editing the repo's
`train-config.yaml` has no effect on a resume.** Every change must go through the
`override` field.

That file is the *resolved* config the original job dumped, so overrides from the
original submission are already baked into it. Do not repeat them — a run
submitted with `--override seed=0` already has `seed: 0` in the dumped config.

`resume.sh` prepends `resume_results.existing_dir=/existing-results` to whatever
`override` holds (`resume.sh:159-160`), so never write that field yourself.

The dataset must contain `config.yaml`, `wandb_run_id`, and
`training_checkpoints/ckpt.tar`; those are what get mounted.

### Changing one entry of a list means replacing the whole list

`override` is a dotlist parsed by `OmegaConf.from_dotlist`
(`fme/core/config.py:28-31`), which has no list-index syntax. Indexing does not
silently misbehave, it fails: `validation.0.evaluate_all_steps=false` builds a
dict keyed `"0"`, and the merge then raises

    ConfigTypeError: Cannot merge DictConfig with ListConfig

So touching one element of `validation` or `inference` means inlining the
**entire** list as a single override value. That works because
`run_gantry_training_job` re-tokenizes `OVERRIDE_ARGS` through `eval`
(`lib.sh:185-193`), so a single-quoted flow-style YAML value survives as one argv
element:

    'validation=[{"name": "val_a", "evaluate_all_steps": false, ...}, {...}]'

Two rules for building one:

- **Generate it from `/existing-results/config.yaml`; never transcribe it.** Real
  values run to thousands of characters. Dump the list, change only the keys you
  mean to change, and assert that reverting them reproduces the original block.
- **Double quotes inside, single quotes outside.** One single quote anywhere in
  the value ends the bash-quoted word and `eval` splits it into garbage.

**`resume.sh` is the only script here that never runs `validate_config`** — the
other six do, so on every other path a bad override is caught before submission.
On a resume nothing checks it and the job fails on the cluster instead. Validate
by hand against the dumped config first:

    python -m fme.ace.validate_config <dumped config> --config_type train \
      --override resume_results.existing_dir=/existing-results <your overrides>

A `--dry-run` does not do this: it prints the parsed fields and the job name, but
never resolves the config.

### Defaults — do these unless there is a reason not to

**Leave `results_dataset` blank.** Blank resolves it at submission time: wandb run
→ `config.environment.BEAKER_EXPERIMENT_ID` → that experiment's `jobs[-1].result`
(`lib.sh:134-138`, `scripts/wandb/wandb_to_beaker_experiment.py`). Because
`resume_wandb=true` overwrites that wandb config key on every resume, the lookup
tracks the newest checkpoint rather than the origin — verified on a run three
resumes deep, where `bt49pyjn` resolves to the last experiment in its chain, not
the first. Pinning a literal dataset id works once and then silently resumes from
a stale epoch the next time the run is extended.

**Keep `tag` unchanged.** The resume is the same seed continuing the same wandb
run, so it should carry the same tag. The resulting Beaker name
`{group}-{tag}-train` (`lib.sh:263-273`) then duplicates the completed
experiment's, which is fine: Beaker auto-appends a 4-char suffix on collision
(`…-rs0-train` and `…-rs0-train-5a53` coexist).

**Set `resume_results.resume_wandb=true`** so the extra epochs continue the
original wandb run instead of starting a second one. This is also what keeps the
`results_dataset` lookup above pointing at the newest checkpoint. The default is
`False` (`fme/core/cli.py:47`).

**Consider `min_runtime`.** It is `0` unless set, meaning preemptible at any time.
On a busy cluster a `high`-priority job can lose many hours to requeue gaps and
redone part-epochs; `1h` or `8h` protects a checkpoint cycle. Auto-resume is
always on for training jobs, so preemption costs time, not progress.

### Check the LR scheduler before raising `max_epochs`

Extending epochs is only clean when no scheduler ties its horizon to
`max_epochs`. A `CosineAnnealingLR` with no explicit `T_max` takes
`T_max = max_epochs` (`fme/core/scheduler.py:37-38`), so raising `max_epochs`
stretches the cosine while the scheduler state is restored from the checkpoint —
the resumed epochs then follow a different LR curve than the original run was on.
`SequentialSchedulerConfig` has the same problem via its `milestones`.

Read `optimization.scheduler` in `/existing-results/config.yaml`. A constant-`lr`
optimizer with no `scheduler` key extends without any of this; anything else needs
a deliberate decision about the LR schedule, recorded alongside the run.

## Reading results

For an experiment that was preempted or retried, always take the **final** job,
never the first — `jq '.[0].jobs[-1].result.beaker'`. Details and the job-status
recipes are in `.cursor/rules/beaker-experiments.mdc` and
`.cursor/rules/check-beaker-job-status.mdc`.

A high job count is normal and not a failure signal. Distinguish the causes in
`jobs[].status.canceledFor`: preemption by a higher-priority job, a cordoned or
unhealthy node, and genuine crashes all look like `exitCode: 1`. Gantry
environment-validation failures exit in under a second, before any Python runs.
Only `jobs[-1].status.exitCode` describes the experiment's outcome.
