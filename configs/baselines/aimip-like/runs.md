# AIMIP-like Baseline Runs

## GPU usage ledger

| Cluster | Priority | Limit (GPUs) | Committed | Active now | Free vs cap |
|---------|----------|--------------|-----------|-----------|-------------|
| Jupiter (ai2/ace) | high | 16 | 9 | 9 | 7 |
| Titan (ai2/climate-titan) | urgent | 4 | 0 | 0 | 4 |

_All jobs are 1 GPU each. **Reconcile cycle (2026-06-11 15:43 → 18:43 UTC, ~3h):** 4 newly finished, all
ec=0/60ep/healthy — **sr0p25-residual-rs0-d471** (Wave 15, xs34zfva, 18:01 UTC, bvl 0.1314, bie 0.0514 —
sr0p25 residual viable; -d471 dual-cluster relaunch confirmed working), **fg8-residual-rs0** (Wave 13,
xnzqzagj, 16:49 UTC, bvl 0.1388, bie 0.0757), **384-residual-rs0** (Wave 13, 1l2rmt0x, 17:20 UTC, bvl
0.1439, bie 0.0749 — **healthy, unlike 256-residual sibling**: the elevated bie 0.1466 on 256-residual is
embed_dim-specific, not residual-architecture-general), **labels-multistep-rs0** (Wave 7, n2utlc8s,
16:10 UTC, bvl 0.1257, **bie 0.0449 — lowest on the board**). No failures this cycle. Jupiter 9 running,
0 queued, **7 free**. **Titan still idle (0/4).** Last reconciled with Beaker: 2026-06-11 18:43 UTC._

_**Wave 17 launched (2026-06-11 14:30 UTC):** 4 new era5-only CRPS/fd-CRPS/
energy-score loss-weight variants (c4d4e2, c5d4e1, c7d2e1, c7d2e1-qsat-scaling) on ai2/ace high
(Jupiter+Titan), commit 2156170de. All four include total_water_path in restricted inference outputs
(commit f553f8ce9, applied to the full submitted config set). Repo cleanup commits in this batch:
5b6988775 (renames to make -lr-tuning explicit in 16 config filenames), f553f8ce9 (+total_water_path on
49 configs), 2156170de (+4 new loss-weight variants), aa6d849aa (run-train.sh Wave 14-17 launch record)._

_**Big throughput cycle (2026-06-10 15:43 → 2026-06-11 14:03 UTC, ~22h):** Jupiter
contention cleared; 7 newly finished (all ec=0, 60 epochs) and 1 newly failed (ec=1) — see breakdown below.
Committed dropped 19 → 11; the entire residual queue drained into running slots and all 11 remaining
committed jobs are now actively running, 0 queued. **Newly finished:** tau200-rs0 (Wave 14,
178gxgv4, 03:51 UTC, bie 0.0636 — healthy, joins tau100 sibling); era5-only-256-residual-rs0 (Wave 13,
15m23cwv, 12:51 UTC, bie **0.1466 elevated** — drift, similar to 384-residual sibling);
no-co2-qsat-scaling-rs0 (Wave 11, g8rapt3y, 10:42 UTC, bie 0.0526 — healthy); qsat-scaling-rs0 (Wave 10,
eiots9f1, 08:04 UTC, bie 0.0525 — healthy); era5-only-c7d2e1-rs0 (Wave 8, uckb3weu, 08:43 UTC, bie 0.0574 —
healthy); labels-fg8-rs0 (Wave 7, 099k85z6, 10:59 UTC, bie 0.0517 — healthy); labels-384-multistep-rs0
(Wave 7, ouk9l16e, 06:49 UTC, bie 0.0585 — healthy). **Newly failed:** era5-only-sr0p50-rs0 (Wave 13,
np7oozsp, 06:35 UTC, ec=1 at ~epoch 20, bvl 0.1528 — investigate). **Titan idle (0/4)** — 4 free urgent
slots; the two -d471 Wave 15 dual-cluster jobs landed on Jupiter, not Titan. Last reconciled with Beaker:
2026-06-11 14:03 UTC._

Status legend: **running** / **queued** (waiting for slot) / **finished** /
**failed** (exited nonzero) / **canceled** (never started, superseded).

## Running / queued (Jupiter+Titan, high, 1 GPU each) — 9 committed (9 running, 0 queued) vs 16 cap

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-c4d4e2-rs0 | era5-only-c4d4e2.yaml | [01KTVH0CPQKJS2HYFERVMJP7YB](https://beaker.org/ex/01KTVH0CPQKJS2HYFERVMJP7YB) | pending | running | Wave 17; era5-only (non-residual) + CRPS/fd-CRPS/energy loss weights 0.4/0.4/0.2; total_water_path in restricted outputs; dual-cluster (Jupiter+Titan) high; launched from 2156170 at 2026-06-11 14:30 UTC |
| train-4deg-daily-v1-era5-only-c5d4e1-rs0 | era5-only-c5d4e1.yaml | [01KTVH0TY5V0EDXGK1WQD6D9GA](https://beaker.org/ex/01KTVH0TY5V0EDXGK1WQD6D9GA) | pending | running | Wave 17; era5-only (non-residual) + CRPS/fd-CRPS/energy loss weights 0.5/0.4/0.1; total_water_path in restricted outputs; dual-cluster (Jupiter+Titan) high; launched from 2156170 at 2026-06-11 14:30 UTC |
| train-4deg-daily-v1-era5-only-c7d2e1-rs0-2156 | era5-only-c7d2e1.yaml | [01KTVH196G07C92HN1HAATKP2Q](https://beaker.org/ex/01KTVH196G07C92HN1HAATKP2Q) | pending | running | Wave 17; era5-only (non-residual) + CRPS/fd-CRPS/energy loss weights 0.7/0.2/0.1 (matches the finished Wave 8 c7d2e1-rs0 but without lr_tuning, lr 0.0001 directly); total_water_path in restricted outputs; dual-cluster (Jupiter+Titan) high; launched from 2156170 at 2026-06-11 14:30 UTC. -2156 suffix because the original c7d2e1-rs0 name belongs to the now-renamed c7d2e1-lr-tuning config |
| train-4deg-daily-v1-era5-only-c7d2e1-qsat-scaling-rs0 | era5-only-c7d2e1-qsat-scaling.yaml | [01KTVH1Q5ZX1M07M8S6YW5684J](https://beaker.org/ex/01KTVH1Q5ZX1M07M8S6YW5684J) | pending | running | Wave 17; era5-only-qsat-scaling base + c7d2e1 loss weights 0.7/0.2/0.1; total_water_path in restricted outputs; dual-cluster (Jupiter+Titan) high; launched from 2156170 at 2026-06-11 14:30 UTC |
| train-4deg-daily-v1-era5-only-qsat-scaling-residual-rs0 | era5-only-qsat-scaling-residual.yaml | [01KTS0D92SPJ7KQS9W06J3H9CP](https://beaker.org/ex/01KTS0D92SPJ7KQS9W06J3H9CP) | pending | running | Wave 16; residual counterpart of the Wave 10 era5-only-qsat-scaling run — era5-only-residual + qsat-scaled shared global-mean removal on specific_total_water_0-7, LHTFLsfc, PRATEsfc, tendency_of_total_water_path_due_to_advection, Q2m; EMA epoch checkpoints on 46-year inference epochs; dual-cluster (Jupiter+Titan) high; launched from a4f8998 at 2026-06-10 12:43 UTC |
| train-4deg-daily-v1-era5-only-residual-rs0 (-d471) | era5-only-residual.yaml | [01KTRNGPCAC23D1TRX7K1SW1B7](https://beaker.org/ex/01KTRNGPCAC23D1TRX7K1SW1B7) | pending | running | Wave 15; rerun of the finished Titan era5-only-residual-rs0 (seed 0), now saving EMA epoch checkpoints on the 46-year inference epochs (1,6,...,56) via ema_checkpoint_save_epochs {start:1,step:5} to recover the well-performing earlier checkpoint. Relaunch with **both clusters (Jupiter+Titan)**; original Jupiter-only -8b9d 01KTPST5 (never started) stopped; launched from d471739 at 2026-06-10 11:44 UTC |
| train-4deg-daily-v1-labels-rollout-rs0 | labels-rollout.yaml | [01KTKR69KY5VS2VGZT4MXKBC4G](https://beaker.org/ex/01KTKR69KY5VS2VGZT4MXKBC4G) | pending | running | Wave 7; non-residual labels + aggressive rollout (50/30/20); launched 13:55 from b49787a |
| train-4deg-daily-v1-labels-multistep-rs1 | labels-multistep-rs1.yaml | [01KTKR76ZY39DT6BNRNCYF9VXF](https://beaker.org/ex/01KTKR76ZY39DT6BNRNCYF9VXF) | pending | running | Wave 7; seed-1 replicate of labels-multistep; launched 13:55 from b49787a |
| train-4deg-daily-v1-labels-c7d2e1-rs0 | labels-c7d2e1.yaml | [01KTM5M39C7E03GTBTWS05H90H](https://beaker.org/ex/01KTM5M39C7E03GTBTWS05H90H) | pending | running | Wave 8; non-residual labels + finite-diff CRPS loss (crps 0.7 / fd-crps 0.2 / energy 0.1), 1-step, fg=1 |

## Finished — Wave 2 (Jupiter, high)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-labels-384-rs0 | labels-384.yaml | [01KTED6W8CSDS1MTKKFEWYG819](https://beaker.org/ex/01KTED6W8CSDS1MTKKFEWYG819) | [vmb9dq6b](https://wandb.ai/ai2cm/ace/runs/vmb9dq6b) | finished | labels 384; completed 60 epochs, finalized 2026-06-07 21:09 UTC ec=0; best_val_loss 0.11894, best_inference_error 0.04981 |
| train-4deg-daily-v1-labels-384-lr-tuning-rs0 (-17ae) | labels-384-lr-tuning.yaml | [01KTENQX6XEPHXF3TX5HB087JG](https://beaker.org/ex/01KTENQX6XEPHXF3TX5HB087JG) | pending | finished | 384 + lr-tuning; completed 60 epochs, finalized 2026-06-07 23:39 UTC ec=0; best_val_loss 0.11620, best_inference_error 0.06036 |
| train-4deg-daily-v1-labels-384-residual-lr-tuning-rs0 (-c1e5) | labels-384-residual-lr-tuning.yaml | [01KTENRBTKNBZK3VG34C4GF21F](https://beaker.org/ex/01KTENRBTKNBZK3VG34C4GF21F) | pending | finished | 384 + residual + lr-tuning; completed 60 epochs, finalized 2026-06-08 01:16 UTC ec=0; best_val_loss 0.11929, best_inference_error 0.05109 |
| train-4deg-daily-v1-era5-only-256-lr-tuning-rs0 (-479a) | era5-only-256-lr-tuning.yaml | [01KTENTM12EM02FZ6V4GJZ3MQR](https://beaker.org/ex/01KTENTM12EM02FZ6V4GJZ3MQR) | pending | finished | 256 embed_dim + lr-tuning; preempted once then resumed; finalized 2026-06-07 05:15 UTC ec=0 |
| train-4deg-daily-v1-era5-only-384-residual-lr-tuning-rs0 (-410a) | era5-only-384-residual-lr-tuning.yaml | [01KTENRT3K1BVEFT7KVAYGSF8T](https://beaker.org/ex/01KTENRT3K1BVEFT7KVAYGSF8T) | pending | finished | era5-only 384 + residual + lr-tuning; finalized 2026-06-07 06:56 UTC ec=0 |
| train-4deg-daily-v1-era5-only-rs1 (-f74a) | era5-only-rs1.yaml | [01KTENS8CFMQW5AQKX1S078REF](https://beaker.org/ex/01KTENS8CFMQW5AQKX1S078REF) | pending | finished | seed 1 replicate; finalized 2026-06-07 07:36 UTC ec=0 |
| train-4deg-daily-v1-era5-only-lr-tuning-rs1 (-98d3) | era5-only-lr-tuning-rs1.yaml | [01KTENSPY7MJ0D0SM58CMH7A53](https://beaker.org/ex/01KTENSPY7MJ0D0SM58CMH7A53) | pending | finished | lr-tuning seed 1 replicate; survived 2 preemptions, resumed 10:12, finalized 2026-06-07 11:29 UTC ec=0 |

## Finished — Wave 3 / 3b (Jupiter, high; residual drift fixes, 20-epoch fine-tunes)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-residual-winds-anomaly-ft-rs0 | era5-only-residual-winds-anomaly-ft.yaml | [01KTEVN65BYNW89MH9P06TTWNY](https://beaker.org/ex/01KTEVN65BYNW89MH9P06TTWNY) | pending | finished | Wave 3; winds anomaly-only residual + multi-step ft |
| train-4deg-daily-v1-era5-only-residual-all-anomaly-ft-rs0 | era5-only-residual-all-anomaly-ft.yaml | [01KTEVNMVC53YKCFJS5QEAWME7](https://beaker.org/ex/01KTEVNMVC53YKCFJS5QEAWME7) | pending | finished | Wave 3; all-variable anomaly-only residual + multi-step ft |
| train-4deg-daily-v1-era5-only-residual-tend-reg-ft-rs0 (-ead2) | era5-only-residual-tend-reg-ft.yaml | [01KTEZ4ZS0R584GYHGN7WBRB7S](https://beaker.org/ex/01KTEZ4ZS0R584GYHGN7WBRB7S) | [snyrjo9b](https://wandb.ai/ai2cm/ace/runs/snyrjo9b) | finished | Wave 3b; relaunch on grad-accum backward fix (21f5400); trained 20 epochs, train 0.106/val 0.156 |
| train-4deg-daily-v1-era5-only-residual-winds-anomaly-tend-reg-ft-rs0 (-37d2) | era5-only-residual-winds-anomaly-tend-reg-ft.yaml | [01KTEZ5E3CCY26SJ3G8EEZE453](https://beaker.org/ex/01KTEZ5E3CCY26SJ3G8EEZE453) | pending | finished | Wave 3b; relaunch on grad-accum backward fix (21f5400) |

## Finished — Wave 5 (Jupiter, high)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-residual-winds-anomaly-tend-reg-rs0 | era5-only-residual-winds-anomaly-tend-reg.yaml | [01KTG12GZMGFW7QRQHDYTTAD25](https://beaker.org/ex/01KTG12GZMGFW7QRQHDYTTAD25) | [zfadhnsj](https://wandb.ai/ai2cm/ace/runs/zfadhnsj) | finished | Wave 5; era5-only control (no labels) to isolate label effect; winds-anomaly+tend-reg (0.05), from-scratch; completed 60 epochs, finalized 2026-06-07 20:30 UTC ec=0; best_val_loss 0.1402, best_inference_error 0.0997 |
| train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg-384-rs0 | labels-residual-winds-anomaly-tend-reg-384.yaml | [01KTG13CBCW8011W8T809TKQ1R](https://beaker.org/ex/01KTG13CBCW8011W8T809TKQ1R) | [jrad94be](https://wandb.ai/ai2cm/ace/runs/jrad94be) | finished | Wave 5; labels + winds-anomaly + tend-reg (0.05) + embed_dim 384, 1-step; completed 60 epochs, finalized 2026-06-08 16:00 UTC ec=0; best_val_loss 0.11986, best_inference_error 35.37 (anomalously high vs siblings — possible inference divergence, worth checking) |
| train-4deg-daily-v1-labels-residual-winds-temp-anomaly-tend-reg-rs0 | labels-residual-winds-temp-anomaly-tend-reg.yaml | [01KTG12YKWBVGT11QQ4WE2QZH1](https://beaker.org/ex/01KTG12YKWBVGT11QQ4WE2QZH1) | [rnpxdv12](https://wandb.ai/ai2cm/ace/runs/rnpxdv12) | finished | Wave 5; labels + winds+temperature anomaly + tend-reg (0.05), 1-step; completed 60 epochs, finalized 2026-06-09 03:52 UTC ec=0; best_val_loss 0.1455, **best_inference_error 58.44 (catastrophic blowup)** — same anomaly-perturbation inference failure as the other winds/all-anomaly residual variants |

## Finished — Wave 7 (Jupiter, high)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-fg8-rs0 | era5-only-fg8.yaml | [01KTKR83EZ0FVG180AM67Y9KYF](https://beaker.org/ex/01KTKR83EZ0FVG180AM67Y9KYF) | [ndy7vp7p](https://wandb.ai/ai2cm/ace/runs/ndy7vp7p) | finished | Wave 7; era5-only (no labels) + filter_num_groups=8, 1-step; label-vs-not contrast at fg8; completed 60 epochs, finalized 2026-06-09 04:43 UTC ec=0; best_val_loss 0.1339, best_inference_error 0.0534 (healthy) |
| train-4deg-daily-v1-labels-fg8-rs0 | labels-fg8.yaml | [01KTKR7N7GZT04H89DP3KANR6Y](https://beaker.org/ex/01KTKR7N7GZT04H89DP3KANR6Y) | [099k85z6](https://wandb.ai/ai2cm/ace/runs/099k85z6) | finished | Wave 7; non-residual labels + filter_num_groups=8, 1-step (vs era5-only-fg8 label contrast); completed 60 epochs, finalized 2026-06-11 10:59 UTC ec=0; best_val_loss 0.11773, best_inference_error 0.0517 (healthy) |
| train-4deg-daily-v1-labels-384-multistep-rs0 | labels-384-multistep.yaml | [01KTKR6RPH4X1QQX1WV4T2J47E](https://beaker.org/ex/01KTKR6RPH4X1QQX1WV4T2J47E) | [ouk9l16e](https://wandb.ai/ai2cm/ace/runs/ouk9l16e) | finished | Wave 7; non-residual labels + multistep + embed_dim 384; completed 60 epochs, finalized 2026-06-11 06:49 UTC ec=0; best_val_loss 0.13307, best_inference_error 0.0585 (healthy) |
| train-4deg-daily-v1-labels-multistep-rs0 | labels-multistep.yaml | [01KTKR5VCCBGHZZDH8R8R9XVHS](https://beaker.org/ex/01KTKR5VCCBGHZZDH8R8R9XVHS) | [n2utlc8s](https://wandb.ai/ai2cm/ace/runs/n2utlc8s) | finished | Wave 7; non-residual labels + multistep rollout (80/15/5); launched 13:55 from b49787a; completed 60 epochs, finalized 2026-06-11 16:10 UTC ec=0; best_val_loss 0.1257, **best_inference_error 0.0449 (lowest on the board)** — multistep + labels combo is the strongest non-residual configuration so far |

## Finished — Wave 8 (Jupiter, high; finite-diff CRPS loss)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-c7d2e1-rs0 | era5-only-c7d2e1.yaml | [01KTM5MHJRF64EABFFFTA5QRNX](https://beaker.org/ex/01KTM5MHJRF64EABFFFTA5QRNX) | [uckb3weu](https://wandb.ai/ai2cm/ace/runs/uckb3weu) | finished | Wave 8; era5-only (no labels) + finite-diff CRPS loss (crps 0.7 / fd-crps 0.2 / energy 0.1), 1-step, fg=1; completed 60 epochs, finalized 2026-06-11 08:43 UTC ec=0; best_val_loss 0.1362, best_inference_error 0.0574 (healthy) |

## Finished — Wave 9 (Jupiter, high)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-no-co2-residual-rs0 | era5-only-no-co2-residual.yaml | [01KTM6CABEHSWAXVR8X14JZWWG](https://beaker.org/ex/01KTM6CABEHSWAXVR8X14JZWWG) | [mgro24pt](https://wandb.ai/ai2cm/ace/runs/mgro24pt) | finished | Wave 9; era5-only residual with global_mean_co2 removed from in_names (CO2 excluded from inputs); 1-step, fg=1; completed 60 epochs, finalized 2026-06-09 07:39 UTC ec=0; best_val_loss 0.1426, best_inference_error 0.0447 (healthy — CO2-excluded residual is fine) |

## Finished — Wave 10 (Jupiter, high; qsat-scaled global-mean removal)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-qsat-scaling-rs0 | era5-only-qsat-scaling.yaml | [01KTMEEFF55QMPVVPKRSH8TYQF](https://beaker.org/ex/01KTMEEFF55QMPVVPKRSH8TYQF) | [eiots9f1](https://wandb.ai/ai2cm/ace/runs/eiots9f1) | finished | Wave 10; era5-only (non-residual) + qsat-scaled shared global-mean removal on specific_total_water_0-7, LHTFLsfc, PRATEsfc, tendency_of_total_water_path_due_to_advection, Q2m; completed 60 epochs, finalized 2026-06-11 08:04 UTC ec=0; best_val_loss 0.1457, best_inference_error 0.0525 (healthy — qsat-scaling does not hurt inference) |

## Finished — Wave 11 (Jupiter, high; qsat + no-CO2)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-no-co2-qsat-scaling-rs0 | era5-only-no-co2-qsat-scaling.yaml | [01KTMEJNG91C4DS2V1AGJZ0ZW9](https://beaker.org/ex/01KTMEJNG91C4DS2V1AGJZ0ZW9) | [g8rapt3y](https://wandb.ai/ai2cm/ace/runs/g8rapt3y) | finished | Wave 11; Wave 10 config + global_mean_co2 removed from in_names (CO2 excluded); completed 60 epochs, finalized 2026-06-11 10:42 UTC ec=0; best_val_loss 0.1449, best_inference_error 0.0526 (healthy — matches qsat-scaling sibling) |

## Finished — Wave 13 (Jupiter, high; embed_dim / spectral_ratio / fg sweeps)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-256-residual-rs0 | era5-only-256-residual.yaml | [01KTPVVFKD6FVSR17SVXSEJB6T](https://beaker.org/ex/01KTPVVFKD6FVSR17SVXSEJB6T) | [15m23cwv](https://wandb.ai/ai2cm/ace/runs/15m23cwv) | finished | Wave 13; era5-only-residual + embed_dim 256; EMA epoch checkpoints on 46-year inference epochs; completed 60 epochs, finalized 2026-06-11 12:51 UTC ec=0; best_val_loss 0.1500, **best_inference_error 0.1466 (elevated)** — drift, similar pattern to the in-progress 384-residual sibling; worth checking against the healthy plain 384-rs0 |
| train-4deg-daily-v1-era5-only-256-rs0 | era5-only-256.yaml | [01KTPVXE69FEQXZ2D8MSM7DJB8](https://beaker.org/ex/01KTPVXE69FEQXZ2D8MSM7DJB8) | [98eiu9he](https://wandb.ai/ai2cm/ace/runs/98eiu9he) | finished | Wave 13; era5-only (non-residual) + embed_dim 256; clean counterpart to the finished 384-rs0; no epoch checkpoints; completed 60 epochs, finalized 2026-06-11 15:41 UTC ec=0; best_val_loss 0.1543, best_inference_error 0.0558 (healthy — non-residual 256 does NOT have the elevated-bie drift seen on the residual 256/384 siblings) |
| train-4deg-daily-v1-era5-only-sr0p25-rs0 | era5-only-sr0p25.yaml | [01KTPVV05NSYFVFXEJC6VJPADT](https://beaker.org/ex/01KTPVV05NSYFVFXEJC6VJPADT) | [ptush1ka](https://wandb.ai/ai2cm/ace/runs/ptush1ka) | finished | Wave 13; era5-only (non-residual) + spectral_ratio 0.25; no epoch checkpoints; completed 60 epochs, finalized 2026-06-11 14:11 UTC ec=0; best_val_loss 0.1344, best_inference_error 0.0512 (healthy — sr0p25 viable, vs sr0p50 sibling failed) |
| train-4deg-daily-v1-era5-only-384-residual-rs0 | era5-only-384-residual.yaml | [01KTPVVZBVBHS97ZDWRJP0RE50](https://beaker.org/ex/01KTPVVZBVBHS97ZDWRJP0RE50) | [1l2rmt0x](https://wandb.ai/ai2cm/ace/runs/1l2rmt0x) | finished | Wave 13; era5-only-residual + embed_dim 384; EMA epoch checkpoints on 46-year inference epochs; completed 60 epochs, finalized 2026-06-11 17:20 UTC ec=0; best_val_loss 0.1439, best_inference_error 0.0749 (healthy — distinct from the 256-residual sibling's elevated bie 0.1466, so the residual-256 drift is embed_dim-specific not residual-general) |
| train-4deg-daily-v1-era5-only-fg8-residual-rs0 | era5-only-fg8-residual.yaml | [01KTPVWYQ8P2Z07K9BGPM758WS](https://beaker.org/ex/01KTPVWYQ8P2Z07K9BGPM758WS) | [xnzqzagj](https://wandb.ai/ai2cm/ace/runs/xnzqzagj) | finished | Wave 13; era5-only-residual + filter_num_groups 8; EMA epoch checkpoints on 46-year inference epochs; completed 60 epochs, finalized 2026-06-11 16:49 UTC ec=0; best_val_loss 0.1388, best_inference_error 0.0757 (healthy) |

## Finished — Wave 4 (Jupiter, high)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-labels-residual-tend-reg-rs0 | labels-residual-tend-reg.yaml | [01KTFSV412E2CWXPBCP5CEHEEA](https://beaker.org/ex/01KTFSV412E2CWXPBCP5CEHEEA) | [kkh6eh27](https://wandb.ai/ai2cm/ace/runs/kkh6eh27) | finished | Wave 4; full residual + tend-reg (0.05) only, 1-step; survived 12 preemptions, completed 60 epochs, finalized 2026-06-09 01:33 UTC ec=0; best_val_loss 0.11752, best_inference_error 0.0506 (healthy — plain residual+tend-reg avoids the anomaly-perturbation inference blowup) |
| train-4deg-daily-v1-labels-residual-all-anomaly-tend-reg-multistep-rs0 | labels-residual-all-anomaly-tend-reg-multistep.yaml | [01KTFSTP81Q6R9E66EPCZADR9Y](https://beaker.org/ex/01KTFSTP81Q6R9E66EPCZADR9Y) | [55kd06ub](https://wandb.ai/ai2cm/ace/runs/55kd06ub) | finished | Wave 4; all-anomaly + tend-reg (0.05) + multistep; completed 60 epochs, finalized 2026-06-09 08:40 UTC ec=0; best_val_loss 0.1991, **best_inference_error 5.59 (elevated blowup)** — multistep softened but did not cure the anomaly-perturbation inference failure |
| train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg-multistep-rs0 (-c936) | labels-residual-winds-anomaly-tend-reg-multistep.yaml | [01KTFST8R1QW9CFYA12SQ9XRVB](https://beaker.org/ex/01KTFST8R1QW9CFYA12SQ9XRVB) | [m3rlv21y](https://wandb.ai/ai2cm/ace/runs/m3rlv21y) | finished | Wave 4; labels + winds-anomaly + tend-reg (0.05) + multistep; completed 60 epochs, finalized 2026-06-09 10:53 UTC ec=0; best_val_loss 0.1276, **best_inference_error 16.85 (blowup)** — winds-anomaly + multistep still fails at inference |

## Finished — Wave 6 (Jupiter, high; relaunches after epoch=None fix 76dc6836d)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-labels-residual-lr-tuning-rs1 (-76dc) | labels-residual-lr-tuning-rs1.yaml | [01KTHFFBCC839PY4J5RDW3YD27](https://beaker.org/ex/01KTHFFBCC839PY4J5RDW3YD27) | [y5tmv4lf](https://wandb.ai/ai2cm/ace/runs/y5tmv4lf) | finished | Wave 6; relaunch of c1bc after epoch=None fix (commit 76dc6836d); completed 60 epochs, finalized 2026-06-10 04:54 UTC ec=0; best_val_loss 0.11458, best_inference_error 0.08810 (fix confirmed — no recurrence of the assert epoch is not None crash) |
| train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg-rs0 (-76dc) | labels-residual-winds-anomaly-tend-reg.yaml | [01KTHFEX200WK2DS8NE70ZNK5Z](https://beaker.org/ex/01KTHFEX200WK2DS8NE70ZNK5Z) | [s4nmhbca](https://wandb.ai/ai2cm/ace/runs/s4nmhbca) | finished | Wave 6; relaunch of 92b1 after epoch=None fix (commit 76dc6836d); completed 60 epochs, finalized 2026-06-10 07:05 UTC ec=0; best_val_loss 0.11656, **best_inference_error 13.99 (blowup)** — fix held (no epoch=None crash), but confirms the winds-anomaly + tend-reg residual variant still diverges at inference, consistent with the other anomaly-perturbation siblings |

## Finished — Wave 15 (Jupiter+Titan, high; dual-cluster relaunches of stalled Wave 12/13 jobs)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-sr0p25-residual-rs0 (-d471) | era5-only-sr0p25-residual.yaml | [01KTRNG7BX8JNQ0JK2CMTK0CMA](https://beaker.org/ex/01KTRNG7BX8JNQ0JK2CMTK0CMA) | [xs34zfva](https://wandb.ai/ai2cm/ace/runs/xs34zfva) | finished | Wave 15; era5-only-residual + spectral_ratio 0.25 (embed_dim 512); EMA epoch checkpoints on 46-year inference epochs. Relaunch with **both clusters (Jupiter+Titan)** to escape Jupiter contention; original Jupiter-only 01KTPVWF (was epoch 2) stopped; launched from d471739 at 2026-06-10 11:44 UTC; completed 60 epochs, finalized 2026-06-11 18:01 UTC ec=0; best_val_loss 0.1314, best_inference_error 0.0514 (healthy — sr0p25 residual viable, matches the sr0p25 non-residual sibling) |

## Finished — Wave 14 (Jupiter+Titan, high; eval-time global-mean relaxation of specific_total_water_0)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-residual-q0-tau100-rs0 | era5-only-residual-q0-tau100.yaml | [01KTQJRZTJ6M1AX6YQ9KYSRJH9](https://beaker.org/ex/01KTQJRZTJ6M1AX6YQ9KYSRJH9) | [57cbbstg](https://wandb.ai/ai2cm/ace/runs/57cbbstg) | finished | Wave 14; era5-only-residual + eval-time global-mean relaxation of specific_total_water_0 toward its normalization mean, e-folding 100 steps; completed 60 epochs, finalized 2026-06-10 15:22 UTC ec=0; best_val_loss 0.14225, best_inference_error 0.06116 (healthy) |
| train-4deg-daily-v1-era5-only-residual-q0-tau200-rs0 | era5-only-residual-q0-tau200.yaml | [01KTQJRG1DJC5PMSW2ZBVXKA6Z](https://beaker.org/ex/01KTQJRG1DJC5PMSW2ZBVXKA6Z) | [178gxgv4](https://wandb.ai/ai2cm/ace/runs/178gxgv4) | finished | Wave 14; era5-only-residual + eval-time global-mean relaxation of specific_total_water_0 toward its normalization mean, e-folding 200 steps (training identical to era5-only-residual baseline); completed 60 epochs, finalized 2026-06-11 03:51 UTC ec=0; best_val_loss 0.14223, best_inference_error 0.0636 (healthy — slightly higher bie than tau100 sibling 0.0612, both well within range) |

## Failed

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-residual-tend-reg-ft-rs0 (orig) | era5-only-residual-tend-reg-ft.yaml | [01KTEVP2QNA479803V5S8020TB](https://beaker.org/ex/01KTEVP2QNA479803V5S8020TB) | [8m68tgnv](https://wandb.ai/ai2cm/ace/runs/8m68tgnv) | failed (ec=1) | Wave 3; tendency reg backward-through-freed-graph bug under grad accumulation; superseded by -ead2 |
| train-4deg-daily-v1-era5-only-residual-winds-anomaly-tend-reg-ft-rs0 (orig) | era5-only-residual-winds-anomaly-tend-reg-ft.yaml | [01KTEVPGS58R3TD4X40HZ555D5](https://beaker.org/ex/01KTEVPGS58R3TD4X40HZ555D5) | pending | failed (ec=1) | Wave 3; same tendency reg bug; superseded by -37d2 |
| train-4deg-daily-v1-labels-384-lr-tuning-rs0 (orig) | labels-384-lr-tuning.yaml | [01KTED6YQZ0506MKV4VWMGBKV8](https://beaker.org/ex/01KTED6YQZ0506MKV4VWMGBKV8) | pending | failed (ec=1) | original 12:06 launch; superseded by -17ae re-launch |
| train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg-rs0 (orig) | labels-residual-winds-anomaly-tend-reg.yaml | [01KTFSAXQX0QVVDCCEV4QK60EF](https://beaker.org/ex/01KTFSAXQX0QVVDCCEV4QK60EF) | pending | failed (ec=1) | Wave 4; config was untracked, not in pushed commit (FileNotFoundError); superseded by -92b1 after commit 8ddd5af |
| train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg-multistep-rs0 (orig) | labels-residual-winds-anomaly-tend-reg-multistep.yaml | [01KTFSBB4WQPS9KPHPKKY7CB6R](https://beaker.org/ex/01KTFSBB4WQPS9KPHPKKY7CB6R) | pending | failed (ec=1) | Wave 4; same untracked-config issue; superseded by -c936 |
| train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg-rs0 (-92b1) | labels-residual-winds-anomaly-tend-reg.yaml | [01KTFSSV0YNPPPNFQ0S4584D1Y](https://beaker.org/ex/01KTFSSV0YNPPPNFQ0S4584D1Y) | pending | failed (ec=1) | Wave 4 primary winds-anomaly run; ran 01:05-07:12 through 3 preemptions, then job 4 crashed ec=1 (canceled=-, genuine) at 07:08. **Same code bug as c1bc**: config enables lr_tuning (epochs.start=1) + a loss schedule (tend-reg); when LR tuning fires at epoch 1, validation batches carry epoch=None → `AssertionError` at loss_schedule.py:45 (trainer.py:391 → lr_tuning.py:121 → single_module.py:1595). Fixed in commit 76dc6836d; superseded by -76dc relaunch |
| train-4deg-daily-v1-labels-residual-lr-tuning-rs1 (-c1bc) | labels-residual-lr-tuning-rs1.yaml | [01KTENT568V8VHMZFQ2ZKG4YP5](https://beaker.org/ex/01KTENT568V8VHMZFQ2ZKG4YP5) | pending | failed (ec=1) | combined seed-1 replicate; survived 2 preemptions, resumed 10:14 then crashed 10:26 ec=1 (canceled=-, genuine). **Code bug, will recur on resume**: `AssertionError: epoch is not None` (loss_schedule.py:45) — LR-tuning validation trial passes epoch=None (single_module.py:1583 → lr_tuning.py:121). Combined lr-tuning + loss-schedule path fixed in commit 76dc6836d; superseded by -76dc relaunch |
| train-4deg-daily-v1-labels-residual-winds-anomaly-tend-reg-rollout-rs0 | labels-residual-winds-anomaly-tend-reg-rollout.yaml | [01KTG13T8CJHHDDFX187EGKJ6Y](https://beaker.org/ex/01KTG13T8CJHHDDFX187EGKJ6Y) | pending | failed (ec=1) | Wave 5; labels + winds-anomaly + tend-reg (0.05) + aggressive rollout (50/30/20); ran through 4 preemptions then crashed 2026-06-08 18:05 UTC ec=1 (canceled=null, genuine) with **same `assert epoch is not None` bug** (lr-tuning + loss-schedule). Launched pre-76dc6836d, so recurs on every resume — **needs relaunch from fixed commit** to complete |
| train-4deg-daily-v1-era5-only-sr0p50-rs0 | era5-only-sr0p50.yaml | [01KTPVTGBZD2Y5AKQ0G07MH2S9](https://beaker.org/ex/01KTPVTGBZD2Y5AKQ0G07MH2S9) | [np7oozsp](https://wandb.ai/ai2cm/ace/runs/np7oozsp) | failed (ec=1) | Wave 13; era5-only (non-residual) + spectral_ratio 0.5; crashed ec=1 at 2026-06-11 06:35 UTC after only ~20 epochs (best_val_loss 0.1528, best_inference_error 0.0600). Sibling sr0p25-rs0 still running — **needs investigation** (loss diverged? OOM? lower priority than re-launch unless reproducible) |

## Canceled / superseded (12:06 batch, never started)

These were the first Jupiter launch; re-launched at 14:35 with `-XXXX` suffixes.

- train-4deg-daily-v1-labels-384-residual-lr-tuning-rs0 ([01KTED718JR83SYA2ADXC6JBDF](https://beaker.org/ex/01KTED718JR83SYA2ADXC6JBDF))
- train-4deg-daily-v1-era5-only-384-residual-lr-tuning-rs0 ([01KTED74HS17V7SQ1V1BPPZX0Y](https://beaker.org/ex/01KTED74HS17V7SQ1V1BPPZX0Y))
- train-4deg-daily-v1-era5-only-rs1 ([01KTED77462MGCF319GRECMCC9](https://beaker.org/ex/01KTED77462MGCF319GRECMCC9))
- train-4deg-daily-v1-era5-only-lr-tuning-rs1 ([01KTED7A62066DFV0NHNJZD09C](https://beaker.org/ex/01KTED7A62066DFV0NHNJZD09C))
- train-4deg-daily-v1-labels-residual-lr-tuning-rs1 ([01KTED7CQF6Q46PNEE08S2S7HQ](https://beaker.org/ex/01KTED7CQF6Q46PNEE08S2S7HQ))
- train-4deg-daily-v1-era5-only-256-lr-tuning-rs0 ([01KTED7F5HRTN8CDTCGF9TEK64](https://beaker.org/ex/01KTED7F5HRTN8CDTCGF9TEK64))

## Canceled / superseded (Wave 15 relaunch, stopped 2026-06-10 11:44 UTC)

Jupiter-only jobs with <5 epochs, stopped and resubmitted dual-cluster (Jupiter+Titan) as `-d471` (see Running / queued).

- train-4deg-daily-v1-era5-only-sr0p25-residual-rs0 ([01KTPVWF4X14ED1CF72Y5HZ2XZ](https://beaker.org/ex/01KTPVWF4X14ED1CF72Y5HZ2XZ)) — was epoch 2; superseded by 01KTRNG7
- train-4deg-daily-v1-era5-only-residual-rs0 (-8b9d) ([01KTPST5BE90QVJWJDG9VF9B90](https://beaker.org/ex/01KTPST5BE90QVJWJDG9VF9B90)) — never started; superseded by 01KTRNGPCA

## Running (Titan) — 0 urgent

_No Titan urgent jobs currently running._

## Finished — Wave 2 (Titan, urgent)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-labels-residual-lr-tuning-rs0 | labels-residual-lr-tuning.yaml | [01KTDDYWA70RMBSJDZ367BRQX0](https://beaker.org/ex/01KTDDYWA70RMBSJDZ367BRQX0) | [88smg5dh](https://wandb.ai/ai2cm/ace/runs/88smg5dh) | finished | Wave 2; all 3 perturbations + ensemble_metrics; completed 60 epochs, finalized 2026-06-07 23:09 UTC ec=0; best_val_loss 0.11517, best_inference_error 0.05178 |
| train-4deg-daily-v1-labels-lr-tuning-rs0 | labels-lr-tuning.yaml | [01KTEC53BG9D2PR00RJSZWMPY2](https://beaker.org/ex/01KTEC53BG9D2PR00RJSZWMPY2) | [29hkpyin](https://wandb.ai/ai2cm/ace/runs/29hkpyin) | finished | Wave 2; labels+lr-tuning; completed 60 epochs, finalized 2026-06-07 23:45 UTC ec=0; best_val_loss 0.11514, best_inference_error 0.06451 |
| train-4deg-daily-v1-era5-only-384-rs0 | era5-only-384.yaml | [01KTED6QFMRQYN4Q0EDFZQKRAB](https://beaker.org/ex/01KTED6QFMRQYN4Q0EDFZQKRAB) | [2hxhchma](https://wandb.ai/ai2cm/ace/runs/2hxhchma) | finished | Wave 2; 384 embed_dim; finalized 2026-06-07 01:22 UTC ec=0 |
| train-4deg-daily-v1-era5-only-384-lr-tuning-rs0 | era5-only-384-lr-tuning.yaml | [01KTED6SQBC9GY0JP77AQVMWNJ](https://beaker.org/ex/01KTED6SQBC9GY0JP77AQVMWNJ) | [ahubrer1](https://wandb.ai/ai2cm/ace/runs/ahubrer1) | finished | Wave 2; 384 embed_dim + lr-tuning; finalized 2026-06-07 00:41 UTC ec=0 |

## Finished — Wave 1 (Titan, urgent)

| Name | Config | Beaker Experiment | wandb ID | Status | Notes |
|------|--------|-------------------|----------|--------|-------|
| train-4deg-daily-v1-era5-only-rs0 | era5-only.yaml | [01KTCT1DWKKX6YHW7R0R9QM60W](https://beaker.org/ex/01KTCT1DWKKX6YHW7R0R9QM60W) | [1brz8cx0](https://wandb.ai/ai2cm/ace/runs/1brz8cx0) | finished | best drift (-0.18K aT7) |
| train-4deg-daily-v1-era5-only-residual-rs0 | era5-only-residual.yaml | [01KTCTN1TCJ89NGQCMTJ9GWC2K](https://beaker.org/ex/01KTCTN1TCJ89NGQCMTJ9GWC2K) | [a0tt7761](https://wandb.ai/ai2cm/ace/runs/a0tt7761) | finished | catastrophic warm drift (motivated Wave 3) |
| train-4deg-daily-v1-era5-only-lr-tuning-rs0 | era5-only-lr-tuning.yaml | [01KTCTNF4DJ6TNVJRGYX2JE28Z](https://beaker.org/ex/01KTCTNF4DJ6TNVJRGYX2JE28Z) | [ivohxxc1](https://wandb.ai/ai2cm/ace/runs/ivohxxc1) | finished | best val loss (0.139) but drift -0.51K |
| train-4deg-daily-v1-labels-rs0 | labels.yaml | [01KTCTMMFENEMJWG54JW0YK51W](https://beaker.org/ex/01KTCTMMFENEMJWG54JW0YK51W) | [y2i0uuny](https://wandb.ai/ai2cm/ace/runs/y2i0uuny) | finished | Wave 1; ran low-priority; finalized 2026-06-07 02:14 UTC ec=0 |
