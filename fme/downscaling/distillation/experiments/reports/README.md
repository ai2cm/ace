# Per-run reports

One report per run, named `<YYYY-MM-DD>-<suffix>-<wandbid>.md`. Generate a
pre-filled report from a wandb run with:

```
conda run -n fme python -m fme.downscaling.distillation.check_runs \
    --report <wandb_id> [--beaker <ULID>] --out fme/downscaling/distillation/experiments/reports/
```

Then write the **Verdict** section by hand. Structure follows
[`../templates/run_report.md`](../templates/run_report.md); eval comparisons follow
[`../templates/eval_report.md`](../templates/eval_report.md). Index of runs is in
[`../LOG.md`](../LOG.md).
