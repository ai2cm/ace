# PRs merged into exp/alexeyfm2

This branch is `main` plus the PRs below. No code changes are made on this
branch outside `configs/experiments/2026-10-19-fm/` and the repo-level
`scripts/` unless they come from a PR listed here.

| PR | Title | Author |
|----|-------|--------|
| [#1514](https://github.com/ai2cm/ace/pull/1514) | Extend torch.compile to every step type and module builder | yyexela |
| [#1519](https://github.com/ai2cm/ace/pull/1519) | Make irfft, NullTimer.child, CappedGELU, and envelope clipping traceable by torch.compile | yyexela |
| [#1520](https://github.com/ai2cm/ace/pull/1520) | Fix SecondaryDecoder.wrap_module mutating the decoder in place | yyexela |
| [#1526](https://github.com/ai2cm/ace/pull/1526) | Recover wandb metrics lost to preemption from on-disk logs | yyexela |
| [#1536](https://github.com/ai2cm/ace/pull/1536) | Add per-group network normalization | yyexela |
| [#1537](https://github.com/ai2cm/ace/pull/1537) | Add get_pooled_stats.py with per-group stats output | yyexela |
| [#1539](https://github.com/ai2cm/ace/pull/1539) | Add float32_matmul_precision option to SingleModuleStepConfig | yyexela |
| [#1541](https://github.com/ai2cm/ace/pull/1541) | Build unconditional modules as if the dataset carried no labels | yyexela |
| [#1543](https://github.com/ai2cm/ace/pull/1543) | Swin U-Net attention and conditioning performance | yyexela |
| [#1544](https://github.com/ai2cm/ace/pull/1544) | Add patch_size, num_levels and skip_projection options to the Swin U-Net | yyexela |
| [#1545](https://github.com/ai2cm/ace/pull/1545) | Pass the network normalizer to GMR and input masking at call time | yyexela |
| [#1549](https://github.com/ai2cm/ace/pull/1549) | Make Samudra and m2lines CappedGELU traceable by torch.compile (via #1514) | yyexela |
| [#1576](https://github.com/ai2cm/ace/pull/1576) | Add output_dir to inference configs to decouple job dir from data output dir | yyexela |
| [#1588](https://github.com/ai2cm/ace/pull/1588) | Stop dataset properties from aliasing member labels and masks | yyexela |
| [#1592](https://github.com/ai2cm/ace/pull/1592) | Add renamed_variables, derived_variables, and variable_metadata to XarrayDataConfig | yyexela |
| [#1597](https://github.com/ai2cm/ace/pull/1597) | Add label-balanced sampling for multi-source training | yyexela |
| [#1600](https://github.com/ai2cm/ace/pull/1600) | Let training skip a bounded run of non-finite-loss batches | yyexela |
| [#1609](https://github.com/ai2cm/ace/pull/1609) | Guard moisture-budget rescale against a zero or near-zero global mean | yyexela |
| [#1610](https://github.com/ai2cm/ace/pull/1610) | Compile every module with CUDA graphs (reduce-overhead) | yyexela |
| [#1611](https://github.com/ai2cm/ace/pull/1611) | Batch EMA weight update with foreach kernels | yyexela |
| [#1612](https://github.com/ai2cm/ace/pull/1612) | Fix Swin padding geometry: consistent CPB latitudes, window-multiple fill, and mirror-mode lat coords | yyexela |
| [#1615](https://github.com/ai2cm/ace/pull/1615) | Add checkpoint weight averaging (model soup) | yyexela |
| [#1617](https://github.com/ai2cm/ace/pull/1617) | Add per-sample global-mean drift inference aggregator | yyexela |
| [#1620](https://github.com/ai2cm/ace/pull/1620) | Add per-dataset spherical harmonic roundtrip of selected variables | yyexela |
| [#1654](https://github.com/ai2cm/ace/pull/1654) | Add per-grid-cell normalization constants | yyexela |

## Branch-local changes

Changes made directly on this branch (not from a PR), outside
`configs/experiments/2026-10-19-fm/`:

- `scripts/data_process/get_pooled_stats.py` gains opt-in config keys:
  `per_grid_cell` (reduce over time only, writing per-cell `centering.nc`,
  `scaling-full-field.nc` and `scaling-residual.nc` on the stores' grid,
  pooled and per group as before; stores naming their horizontal dims
  differently, e.g. grid_yt/grid_xt vs lat/lon, are matched by dim position and
  coordinate values), `scalar_variables` (kept scalar in per-cell
  mode), `scalar_reference` (copy those scalars from an existing stats
  directory, per group, so they equal a baseline's constants), `pooling:
  label_equal` (root stats give each group equal total weight, timestep-weighted
  within a group; a variable missing from a group is pooled over the others
  with renormalized equal weights) and `group_placeholders` (per-cell mean 0 /
  std 1 for variables a group lacks). Defaults are unchanged; tests in
  `scripts/data_process/test_get_pooled_stats.py`.
- `scripts/data_process/configs/fm-v3-e3sm-balanced-percell-stats.yaml`: the
  v2 fm-e3sm-balanced stats config read from weka, with per-cell stats,
  label-equal pooling, HGTsfc / land, ocean and sea-ice fractions /
  global_mean_co2 copied from the v2 stats, and TMP850 / h500 placeholders in
  the e3sm group. Checked by
  `configs/experiments/2026-10-19-fm/scripts/verify_percell_stats.py`.
- `scripts/data_process/configs/fm-v3-e3sm-balanced-percell-c3-stats.yaml`: the
  same stores grouped per dataset configuration (era5, amip, ramped, som,
  cm4-picontrol, cm4-1pctco2, am4-randco2, xshield, e3sm) for the C3 runs, which
  use only its groups; computed from the per-store stats of the config above.

## Deferred

These PRs are open as drafts and are not merged into this branch.

| PR | Title | Author |
|----|-------|--------|
| [#1618](https://github.com/ai2cm/ace/pull/1618) | Drift-gated soup checkpoint selection in trainer | yyexela |

## Anneal sweep

30 fine-tuning runs (6 arms x 5 seeds) from the balanced nc-sfno parents'
`ema_ckpt_0150.tar`, wandb group `ace2-fm-anneal-2026-10-19`. Configs, arms,
deviations, log keys, smoke results and Beaker ids:
[run_configs/anneal/README.md](run_configs/anneal/README.md). Generated by
`create_anneal_jobs.py`, submitted by `submit_anneal_jobs.py`.
Post-processing (README "Post-processing"): `make_soup_checkpoints.py`
(checkpoint C, soup of `ema_ckpt_0016..0020` -> Beaker `{run}-ckptC`,
`anneal_soups.json`), `submit_anneal_eval_jobs.py` (J1/J3/J4 for checkpoints
C, B, P), `run_anneal_postprocess.sh` (map refresh + soups + evals, run once
training is done).

## FM v2 runs (2026-11-09)

24 training runs from scratch, wandb group `ace2-fm-2026-11-09` (ai2cm/FM):
2 architectures of about equal parameter count (`nc-sfno-v2`: SFNO embed_dim
1024, 52.6M params; `nc-swin-v2.2-v2`: Swin embed_dim 224, 50.9M params) x 4
data arms x 3 seeds, masking off, 4 GPUs (global batch 8). Jobs are
`ace2-fm-nc-{sfno,swin-v2.2}-v2-fm-{era5,balanced,e3sm-balanced,spencer-balanced}-s{0..2}-maskoff`;
s0 is submitted at Beaker priority high, s1/s2 at normal. `fm-era5` is the
ERA5 training data alone; `fm-spencer-balanced` is exactly the balanced
parents' data (era5, c96, cm4, xshield) with their v2 stats; `fm-balanced` adds
c96 AMIP ic_0002 and c96 SOM ic_0002..0005 (except held-out 1xCO2-ic_0005);
`fm-e3sm-balanced` adds E3SMv3 piControl to fm-balanced. fm-era5, fm-balanced
and fm-e3sm-balanced use stats pooled over exactly their own training data
(`/climate-default/alexeyy/norm_ablation_0/2026-11-09-fm-v2-*`). Labels are
sampled with equal weights, 2758 samples (time_buffer windows) per epoch in
every arm, the fm-era5 default. 50 epochs: 1 forward step for epochs 1-40, 3
steps (loss on every step, gradients detached) for 41-50; LR 1e-4 constant
(swin keeps a 2-epoch warmup); EMA 0.9999 with EMA checkpoints every epoch;
HGTsfc roundtripped (PR #1620) on cm4- and xshield-labelled training data.
`best_ckpt` is selected on ERA5 validation (1994, 2014) alone, with a weight-0
`val_c96` logged in the non-ERA5 arms; `best_inference_ckpt` on
aimip_checkpoint_era5 (every epoch) alone; the ERA5 10yr/46yr runs go every
10 epochs. Generated by `create_fm_v2_jobs.py` (its docstring lists every
change from the parents), submitted by `submit_fm_v2_jobs.py`. Progress rows
in `fm_table.py`.

Multi-step training runs under fme's default `reduce-overhead` compile mode
(CUDA graphs) thanks to PR #1610, which clones gradients accumulated across
forward steps out of the CUDA-graph pools; before it, multi-step backward
failed with "accessing tensor output of CUDAGraphs that has been overwritten
by a subsequent run".

## FM v3 runs (2026-11-18)

18 training runs from scratch testing per-grid-cell normalization (PR #1654),
wandb group `ace2-fm-sfno-v3-2026-11-18` (ai2cm/FM): 3 network normalizations
x 2 loss normalizations x 3 seeds, each identical to the v2
`nc-sfno-v2-fm-e3sm-balanced` maskoff run of its seed (SFNO embed_dim 1024,
4 GPUs, global batch 8, same data, sampling, EMA, loss, validation and inline
inference) except for the recipe and normalization. Jobs are
`ace2-fm-nc-sfno-v3-fm-e3sm-balanced-{c1,c2,c3}-{scalarloss,cellloss}-s{0,1,2}-maskoff`;
s0 at Beaker priority high, s1/s2 normal.

- Recipe: 120 epochs; 1 forward step for epochs 1-114, 3 (loss on every
  step, gradients detached) for 115-120; LR 1e-4 constant with no warmup,
  then ext80's 1-sqrt cooldown to 0 over epochs 115-120 (SequentialLR stepped
  per iteration, 3105 iterations per epoch). Heavy inline inference every 10
  epochs.
- Network normalization: `c1` pooled per-cell from STATS_DIR
  (`/climate-default/agents/alexeyy/agent-alexeyy-fm-v3-percell-stats-75w7/2026-11-18-fm-v3-e3sm-balanced-percell`,
  per-cell on the 45 x 90 grid, HGTsfc, land/ocean/sea-ice fraction and
  global_mean_co2 scalar as in v2; std floor 0.01); `c2` the same plus
  per-cell groups c96/era5/cm4/xshield/e3sm from STATS_DIR/groups (e3sm with
  TMP850/h500 placeholders); `c3` the same pooled block plus per-cell groups
  per config from
  `/climate-default/agents/alexeyy/agent-alexeyy-fm-v3-percell-c3-stats-legy/2026-11-18-fm-v3-e3sm-balanced-percell-c3/groups`
  (era5, amip, ramped, som, cm4-picontrol, cm4-1pctco2, am4-randco2,
  xshield, e3sm), with the training data relabelled to those 9 labels
  (labels drive conditioning too; `val_c96` becomes `amip`) and each source's
  0.2 sampling weight split across its labels by training timesteps.
  Pinned variables and default label (era5) as in v2.
- Loss normalization: `scalarloss` the v2 scalar stats
  (`2026-11-09-fm-v2-e3sm-balanced` centering + full-field, residual for
  prognostics), exactly the v2 loss; `cellloss` pooled per-cell STATS_DIR
  centering + full-field / residual with std floor 0.1.
- Generated by `create_fm_v3_jobs.py` (its docstring lists every change from
  v2), checked on CPU by `validate_fm_v3_configs.py`, submitted through the
  agent VM's gantry shim by `submit_fm_v3_jobs.py`. Jobs run
  `fme.ace.train --override experiment_dir=$AGENT_OUTPUT_DIR`, so checkpoints
  live under `/climate-default/agents/alexeyy/<experiment>/training_checkpoints`
  and survive preemption; experiment names, Beaker ids and checkpoint paths
  are in `records/fm_v3_jobs.json`. Progress rows in `fm_table.py`.
- Smoke tests (`...-c2-cellloss-s0-maskoff-smoke`, `...-c3-cellloss-s0-maskoff-smoke`,
  group `ace2-fm-sfno-v3-2026-11-18-smoke`: 2 epochs of 288 iterations, epoch 2
  3-step with the cooldown) passed under the default reduce-overhead compile.
- Requeue incident (2026-11-18): 10 s1/s2 runs (c1-cellloss, c2-cellloss,
  c2-scalarloss, c3-cellloss, c3-scalarloss) were stopped and resubmitted
  with ai2/saturn added on the wrong belief that they had never started.
  Each had trained 28-30 epochs and was queued again after a preemption
  (wandb runs now "crashed"); the resubmissions had new experiment names,
  hence empty `$AGENT_OUTPUT_DIR`s, and would have restarted from epoch 0.
  They were replaced before starting by jobs submitted with
  `submit_fm_v3_jobs.py --resume-from <original experiment>`, whose wrapper
  copies the original experiment dir (checkpoints, `wandb_run_id`, metrics;
  read-only on the source) into the job's own `$AGENT_OUTPUT_DIR` behind a
  completion marker, so fme resumes from the copied `ckpt.tar` and the
  original wandb run. Each record's `resumed_from` and `superseded` give the
  chain.

## C96 AMIP-only runs (2026-11-18)

12 training runs from scratch, wandb group `ace2-fm-c96amip-2026-11-18`
(ai2cm/FM): `ace2-fm-{nc-sfno-v2,nc-swin-v2.2-v2}-fm-{c96amip,era5-1979}-s{0,1,2}-maskoff`,
s0 at priority high, s1/s2 normal, on jupiter/titan/saturn. Each is the v2
`fm-balanced` maskoff run of its architecture and seed (Swin at the v2 size
with its 2-epoch warmup) with the FM v3 recipe (120 epochs, 3-step with the
1-sqrt cooldown for 115-120), trained on 1979-1993 and 1995-2008 only:

- `c96amip`: C96 AMIP ic_0001 (label `c96`, 3105 iterations per epoch),
  scalar stats pooled over that data (`fm-c96amip-stats.yaml`,
  `/climate-default/agents/alexeyy/agent-alexeyy-fm-c96amip-stats-gpu-0isd/2026-11-18-fm-c96amip`,
  checked by `verify_c96amip_stats.py`). `best_ckpt` on `val_c96` (1994,
  2014), `best_inference_ckpt` on a new `aimip_checkpoint_c96` (the
  aimip_checkpoint_era5 entry on the C96 AMIP store); ERA5 entries kept at
  weight 0. Every dataset is labelled `c96`, else fme would feed an
  all-zero label vector.
- `era5-1979`: ERA5 over the same years (label `era5`), its own stats
  (`fm-era5-1979-stats.yaml`,
  `/climate-default/agents/alexeyy/agent-alexeyy-fm-era5-1979-stats-gpu-c1ic/2026-11-18-fm-era5-1979`)
  and v2 fm-era5 selection, C96 entries at weight 0. No smoke test of its own.
- Generated by `create_fm_c96amip_jobs.py` (docstring lists every change
  from v2), checked by `validate_fm_c96amip_configs.py`, submitted by
  `submit_fm_c96amip_jobs.py --arm ...`; records in
  `records/fm_c96amip_jobs.json`, progress rows in `fm_table.py`.
- C96 smoke tests (group `ace2-fm-c96amip-2026-11-18-smoke`) passed on
  jupiter after the saturn-only first pair stayed queued and was replaced.
  The stats jobs ran on 1-GPU nodes because CPU-only submissions never left
  "initializing".
