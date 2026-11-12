# FM v2 SFNO: 2-GPU diagnostic (2026-11-09)

## Purpose

v2 SFNO grid runs use 4 GPUs (`N_GPUS = "4"` in `scripts/submit_fm_v2_jobs.py`, no
recorded rationale). This tests whether 2 GPUs fits in memory and how much slower
it is, per GPU type. Setup is unchanged from the grid: global batch 8,
n_ensemble 2, embed_dim 1024, `use_gradient_accumulation: true`, torch.compile on.

## Config

`run_configs/ace-train-config-4deg-nc-sfno-v2-fm-balanced-s0-maskoff-2gpu-diag.yaml`
(commit 9aafe168f) is a copy of `...-fm-balanced-s0-maskoff.yaml` with 3 edits:

1. `max_epochs: 50` -> `2`
2. the four every-10 inline-inference blocks `start: 9` -> `start: 0`, so the
   10-year and 46-year rollouts and the weather ensembles run after epoch 0
3. the `n_forward_steps` milestone `epoch: 41` -> `epoch: 1`

**Correction (2026-11-10).** Edit 3 was meant to make only e1 a 3-step epoch, but
milestone epochs are 1-indexed: the trainer calls `set_epoch(epochs_trained + 1)`,
so the first epoch is epoch 1. With `epoch: 1`, **both e0 and e1 trained with 3
forward steps**. wandb confirms it: the 2-GPU runs log `loss_step_0/1/2` from the
first epoch, and their total loss is the sum over the 3 steps (0.276 + 0.359 +
0.410 = 1.045 after e0). That is why their loss looked about 4x the baseline's. Per
step the models are comparable: `loss_step_0` 0.276 vs 0.244. In the grid,
`epoch: 41` means epochs 41-50 in 1-indexed terms (e40-e49 in this doc's 0-indexed
notation) train with 3 steps. Earlier versions of this doc compared the 2-GPU 3-step
e0 against the 4-GPU 1-step e0, so the 2.03x slowdown and the conclusions built on
it were wrong. Everything below uses matched step counts.

Submitted with N_GPUS=2, high priority, wandb group `ace2-fm-2026-11-09-diag`
(project ai2cm/FM).

## Results

| Cluster / GPU | Job | Outcome | Peak mem | Train e0, 3 steps (s, incl. compile) | Train e1, 3 steps (s) | Heavy inference (s) |
|---|---|---|---|---|---|---|
| titan / B200 192.3 GB | 01M71D4D8N2YE7ZACBFJ209YZN (wandb yupivt4b) | exit 0, no OOM | 186.0 GB (96.8%) | 870 | 737 | 636 |
| jupiter / H100 85.5 GB* | 01M71D4PEHCP0SR8TPM42MC401 (wandb n0hfkx2s) | exit 0, no OOM | 71.8 GB (84.0%) | 1232 | 1111 | 715 |
| saturn / A100 80 GB | 01M71D4YJDCX31Q5M2YAP924G8 | **NOT RUN**: still queued after 6 h (team cap / cluster full) | - | - | - | - |

\*Total memory as reported by wandb (bytes / percent).

Neither log contains `out of memory`, `CUDA error` or `Traceback`.

Other phases (titan / jupiter):

- Validation: e0 180 / 173 s, e1 49 / 52 s.
- Heavy inference after e0, by rollout: 46 windows 69 / 80 s, 92 windows
  121 / 138 s, 420 windows 441 / 494 s.
- Light inference after e1 (46 windows): 92 / 95 s.
- Peak memory per phase on jupiter: e0 train 48.6 GB, e0 validation 68.8 GB,
  e0 inference 68.3 GB, e1 train 70.5 GB, e1 validation and inference 71.8 GB.
  Both train epochs are 3-step, so the e0/e1 train difference is not step count.

Memory is the PyTorch-cached high-water mark (wandb `memoryAllocated`), so it is an
upper bound on what the run needs. The caching allocator expands to fill the card.
Titan reached 96.8% at 2 GPUs, and the 4-GPU baseline reached 92.2%, but the same
2-GPU workload completed on H100 with 71.8 GB. So the titan percentage says
nothing about whether the run fits on an 80 GB card. Memory was measured only with
3 steps, the worst case.

## Comparison with the 4-GPU titan baseline

Baseline: job 01M71AR66XW5A9BSCJ7CE3EMSV, wandb x4f0rp2e, B200. It completed all 50
epochs: e0-e39 with 1 step, e40-e49 with 3 steps. Both runs take ~3100 optimizer
steps per epoch, because the global batch is fixed: per-rank batch 2 at 4 GPUs, 4 at
2 GPUs. Train times come from the job logs ("Starting training step" to "Starting
validation step").

| Train epoch (titan) | 4 GPU | 2 GPU | 2 GPU / 4 GPU time | GPU-s, 4 GPU | GPU-s, 2 GPU |
|---|---|---|---|---|---|
| 1 step, steady (s) | 298 (median e1-e39, range 282-318) | **not measured** | - | 1192 | - |
| 3 steps, steady (s) | 604 (median e40-e49, range 562-619) | 737 (e1) | **1.22x** | 2416 | 1474 |
| 3 steps, s per 100 steps | 17.2 | 21.5 | 1.25x | | |
| 1 step, s per 100 steps | 7.4 | - | - | | |
| Validation (s) | 185 (e0) | 180 (e0), 49 (e1) | ~1.0x | | |
| Heavy (every-10) inference (s) | ~600 | 636 | ~1.06x | | |

- At 3 steps, halving the GPU count slows training only 1.22x and saves ~39% of
  GPU-seconds. The 4-GPU run does not use its GPUs fully (e0 GPU busy 34%, power
  38% of cap), so each GPU does more useful work at per-rank batch 4.
- Validation and inference take the same wall time at either GPU count, so they
  also cost about half the GPU-seconds at 2 GPUs.
- 1-step numbers at 1, 2 and 4 GPUs come from the GPU-count sweep below.
- Gap: n=1 per config, on different nodes.

## Ablation: training-loader workers (2 GPU, titan)

Config: `run_configs/ace-train-config-4deg-nc-sfno-v2-fm-balanced-s0-maskoff-2gpu-workers8-diag.yaml`
(commit 6aa90b123), identical to the 2-GPU diag config except
`train_loader.num_data_workers: 4` -> `8`. Job 01M73EJSTP1QMAG2F2YM8Z2T2T
(wandb 8jd4w5ow), titan B200: exit 0, no `out of memory` / `CUDA error` / `Traceback`.
Both 2-GPU runs train with 3 steps in both epochs, so this comparison is matched.
The original motivation compared against the 4-GPU 1-step e0, which is not a
matched comparison (see the correction above).

| | 2 GPU, 4 w/rank (yupivt4b) | 2 GPU, 8 w/rank (8jd4w5ow) |
|---|---|---|
| e0 train (s, incl. compile) | 870 | 2216 |
| e1 train (s) | 737 | 865 |
| e0 GPU busy / power | 66% / 67% | 48% / 39% |
| e1 GPU busy / power | 74% / 72% | 66% / 68% |
| Peak mem (e0 / e1) | 96.0% / 96.4% | 98.7% / 99.1% |
| Time to step 100 (s, compile) | 164 | 584 |
| Median s per 100 steps, e0 | 21.8 | 39.8 (7 of 29 intervals >1.5x median) |
| e0 validation / heavy inference (s) | 180 / 636 | 242 / 698 |
| Val loss after e0 / e1 | 1.041 / 0.910 | 1.040 / 0.910 |

GPU stats are means over all GPUs within each train window (wandb system metrics);
memory is the peak.

Conclusion: 8 workers per rank did not speed up training. It was slower in every
phase, including validation and inference, whose loaders did not change (+34% and
+10%). e0 had long stalls, and by the end of e0 and in e1 the run settled at ~23-24 s
per 100 steps, close to the 4-worker run. The extra slowness is most likely node or
storage variance (different node, possibly a cold read cache) or worker
oversubscription; one run cannot separate those. Keep `num_data_workers: 4`.
Gap: n=1 per config, different nodes.

## GPU-count sweep (2026-11-11)

Goal: GPU-hours per 50-epoch run at each GPU count. One config for all runs,
`run_configs/ace-train-config-4deg-nc-sfno-v2-fm-balanced-s0-maskoff-gpu-sweep-diag.yaml`
(commit 8e356c13d), identical to the grid config except:

1. `max_epochs: 50` -> `3`
2. `n_forward_steps` milestone `epoch: 41` -> `epoch: 3`, so e0 is 1-step with
   compile, e1 is 1-step steady state and e2 is 3-step steady state
3. the four every-10 inline-inference blocks `start: 9` -> `start: 0`, so heavy
   inference runs once, after e0

Normal priority, wandb group `ace2-fm-2026-11-09-diag`. All four exited 0, with no
`out of memory`, `CUDA error` or `Traceback` in the logs. The 4-GPU job waited
~25 h in the queue, the others 10-13 h (all were queued behind the
team's own grid runs).

| Run | Job | wandb |
|---|---|---|
| 1 GPU titan (B200) | 01M73QP0QFCWW16BPXR05FR1KZ | 0hw4lf93 |
| 2 GPU titan (B200) | 01M73QNS608ERSZWSS9Z5G5FWD | kfm40cfd |
| 4 GPU titan (B200) | 01M73QP7YEH0FG5EPAN0HZJB2X | 2ssj1xa6 |
| 2 GPU jupiter (H100) | 01M73QPF3CS5C05GPAESKTHQVW | fubsixc2 |

Phase wall times (s) from the job logs. Validation and light inference are the
mean of e1 and e2. Heavy inference is the post-e0 inference time minus one light
inference. One-off is the extra time of e0 training (compile) and e0 validation
over steady state.

| Phase | 1 GPU titan | 2 GPU titan | 4 GPU titan | 2 GPU jupiter |
|---|---|---|---|---|
| Train, 1 step (e1) | 593 | 340 | 275 | 450 |
| Train, 3 steps (e2) | 1297 | 779 | 581 | 1127 |
| s per 100 steps, 1 / 3 steps | 12.6 / 38.2 | 8.3 / 22.4 | 6.7 / 16.3 | 12.3 / 33.7 |
| Validation | 52 | 25 | 31 | 26 |
| Light inference | 146 | 118 | 108 | 122 |
| Heavy inference | 978 | 564 | 452 | 648 |
| One-off (compile, first validation) | 322 | 302 | 248 | 258 |
| Peak memory (cached high-water) | 187.9 GB (97.7%) | 186.0 GB (96.7%) | 177.5 GB (92.3%) | 71.7 GB (83.9%) |

Projected 50-epoch run: wall = 40 x train_1step + 10 x train_3step + 50 x
validation + 50 x light + 5 x heavy + one-off (light inference runs every epoch,
heavy every 10th). GPU-hours = N_GPUs x wall.

| | 1 GPU titan | 2 GPU titan | 4 GPU titan | 2 GPU jupiter |
|---|---|---|---|---|
| Wall time per run (h) | 14.4 | 8.8 | 7.3 | 11.2 |
| of which training (h) | 10.2 | 5.9 | 4.7 | 8.1 |
| **GPU-hours per run** | **14.4** | **17.6** | **29.2** | **22.3** |
| GPU-hours vs 4 GPU titan | 0.49x | 0.60x | 1.00x | - |
| Runs in parallel under a 32-GPU cap | 32 | 16 | 8 | 16 |

- Check against a real run: the projection gives 7.3 h for 4 GPUs on titan. The
  4-GPU baseline x4f0rp2e took 7.7 h from start to exit, including setup and a
  slightly slower node (298 vs 275 s per 1-step epoch).
- Node variance: the 4-GPU control ran 1-step epochs 8% faster than the baseline,
  and the 2-GPU titan 3-step epoch took 779 s here vs 737 s in the first diag.
  Treat differences under ~10% as noise (n=1 per config).
- The GPUs are underused at small per-GPU batches. Going from 4 GPUs to 1
  multiplies 1-step training time by only 2.2x, not 4x, so per-GPU throughput
  more than doubles. Validation and inference barely speed up with more GPUs,
  so extra GPUs sit mostly idle through them.
- Peak memory is the PyTorch cache high-water mark and fills the card on B200 at
  every GPU count, so it does not show how much a run needs. The only
  80 GB-class measurement is 2 GPUs on H100: 71.7 GB at 4 samples per GPU.
  1 GPU means 8 samples per GPU, which very likely does not fit on an 80 GB
  card. That is untested.

## Recommendation

**titan (B200): use 1 GPU for queue-bound grids, 2 GPUs when per-run latency
matters. Do not use 4.**

- 1 GPU costs 14.4 GPU-hours per run, half of 4 GPUs (29.2). Under the cap it runs
  4x as many jobs at once as 4 GPUs. Each run takes 14.4 h instead of 7.3 h, so a
  large grid finishes about 2x sooner overall. Single-GPU jobs also fit any free
  slot, while the 4-GPU control waited ~25 h for a slot.
- 2 GPUs cost 17.6 GPU-hours (0.60x of 4 GPUs) and take 8.8 h per run, only 1.2x
  slower than 4 GPUs. Use 2 GPUs for runs needed soon, such as a single
  re-run.
- 4 GPUs save 1.5 h per run over 2 GPUs and cost 66% more GPU-hours.

**jupiter (H100): use 2 GPUs.** 22.3 GPU-hours and 11.2 h per run, with a 71.7 GB
peak. 1 GPU very likely does not fit in 80 GB, and 4 GPUs on H100 was not
measured; on titan, 4 GPUs cost 1.66x the GPU-hours of 2.

**saturn (A100 80 GB): use 2 GPUs.** It is untested, but the 71.7 GB H100 peak is
below 80 GB. A100 is slower than H100, so expect longer runs than on jupiter.

To apply: `N_GPUS` in `scripts/submit_fm_v2_jobs.py` is a single value for all
clusters, so per-cluster GPU counts need one submission per cluster (or a
cluster-to-GPU-count map in the script). Keep `num_data_workers: 4` (see the
ablation). The global batch stays 8 at every GPU count, so training is the same
apart from DDP numerics.
