# SamudrACE CM4-piControl paired inference jobs

Two Beaker/gantry jobs that run the same SamudrACE CM4-piControl inference,
each the other's comparison counterpart:

- `e2s/` — earth2studio-side job, running the SamudrACE prognostic wrapper
  from the earth2studio fork branch. See
  [the wrapper PR](https://github.com/jpdunc23/earth2studio/pull/1)
  (TODO: replace with the upstream earth2studio PR once opened).
- `fme/` — fme reference-side job, running vanilla coupled inference at a
  pinned ace tag.

Details (what each runs, outputs, environment) are in `e2s/README.md` and
`fme/README.md`.

## Submitting

```bash
bash e2s/submit.sh
bash fme/submit.sh
```

## Varying an experiment

The two sides are varied differently:

- `e2s/`: everything is set by environment variables — artifact dataset,
  scenario, initial condition, cycle count. See `e2s/submit.sh` for the
  variable names and defaults.
- `fme/`: edit `fme/inference-config.yaml`; only the artifact dataset is
  env-overridable. See `fme/submit.sh`.

**Warning:** each `submit.sh` hardcodes its gantry `--name`/`--description`
strings (the fme side's description restates values that really come from
`inference-config.yaml`). If you vary an experiment, edit those strings too
or the Beaker job metadata will lie.

These scripts diverge from the commit-and-push convention in
`configs/README.md`: instead of running gantry from this repo's current
commit, each `submit.sh` clones a pinned ref into a temp dir itself.

## Last validated

Both jobs succeeded on GPU on 2026-07-30. Pins that will age:

- the torch wheel pinned for the cluster's driver era in each `submit.sh`
- gantry's `--min-runtime` flag
- the ace tag (`ACE_TAG` in `fme/submit.sh`)
