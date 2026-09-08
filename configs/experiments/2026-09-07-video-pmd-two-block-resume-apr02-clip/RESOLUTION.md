# Resolution — NOT NEEDED, the store was already complete

The original global two-block inference job (experiment
`01M1HFBRC65YMQ66YDPDHAZ8XX`) exited 1 on a final-`dist.barrier()` NCCL
timeout: ranks 1-3 finished all 92 clips, hit the barrier, and timed out after
30 min waiting for rank 0 (a ~40-min straggler over the 5-day run). Rank 0's
`out.log` snapshot in the result dataset stops at "batch 91/92 written", which
looked like a missing final clip.

**It wasn't.** A read-only full-store NaN scan
(`../../../scratch scan_gaps.py` equivalent, 2026-09-08) found:

```
eastward_wind_at_ten_meters:    0 steps ANY nan, 0 ALL nan
northward_wind_at_ten_meters:   0 steps ANY nan, 0 ALL nan
PRMSL:                          0 steps ANY nan, 0 ALL nan
PRATEsfc:                       0 steps ANY nan, 0 ALL nan
air_temperature_at_two_meters:  0 steps ANY nan, 0 ALL nan
*** NO NaN ANYWHERE — the store is complete. ***
```

Rank 0's batch 92 write *did* complete in the ~68 min between its last log
line and the SIGABRT cascade; only the "Completed inference" line and the
per-rank progress logs for ranks 1-3 (silenced — the logger only prints on
rank 0) never showed. The output
`…/two-block-coarse-endpoints-flat/test-2023-2024-ens4-global.zarr` is fully
usable as-is; the exit-1 was purely the barrier.

The `merge_apr02_clip.py` safety assert (target region must be all-NaN before
writing) caught this and aborted without touching the main store. The scratch
`RESUME-2023-04-02-clip.zarr` was deleted from weka.

**Kept for reference / reuse:** if a future patch-tiled `video_inference`
run genuinely dies mid-way, `video_inference.yaml` (narrow `data.subset` +
`batch_size: 1` + `max_batches`) + `merge_apr02_clip.py` (zarr `mode="r+"`
region copy with all-NaN precondition) is the pattern to targeted-resume one
clip without re-running the whole job or overwriting the store.

**Also worth fixing for real:** the barrier timeout. The trailing
`dist.barrier()` in `video_inference.run_inference` only gates the
"Completed inference" log line — ranks write to disjoint time regions — so a
longer process-group timeout, or dropping that barrier, or even sharding, all
prevent a near-complete multi-day run from exiting 1.
