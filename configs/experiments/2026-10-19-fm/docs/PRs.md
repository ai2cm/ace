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
