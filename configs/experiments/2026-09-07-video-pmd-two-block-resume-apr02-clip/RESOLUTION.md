# Resolution — the Apr 2 clip WAS missing (zero-filled, not NaN-filled)

The original global two-block inference job (experiment
`01M1HFBRC65YMQ66YDPDHAZ8XX`) exited 1 on a final-`dist.barrier()` NCCL
timeout: ranks 1-3 finished all 92 clips, hit the barrier, and timed out
after 30 min waiting for rank 0 (a ~40-min straggler). Rank 0's `out.log`
stops at "batch 91/92 written" — its batch 92, the
**2023-04-02 00:00 → 21:00 window (8 frames)**, was never written.

## First pass got this wrong

A full-store NaN scan (2026-09-08, first attempt) found no NaN and concluded
"store is complete". **That was wrong.** `fme/core/writer.py`'s ZarrWriter
initializes arrays with `fill_value=0.0`, not NaN — so an unwritten clip
reads back as **all zeros**, which a NaN-only scan misses, and which made
this script's original all-NaN precondition abort with "target region is all
finite" (zeros are finite).

The real symptom surfaced in `crps_eval.py`: two-block's global PRMSL RMSE
came out at **286 mb** and T2m at **81 K** — physically impossible. A
per-latitude-band diagnostic showed 2023-04-02 12:00's entire global field is
`[0.0, 0.0]` for PRMSL and T2m (fine on Jan 2 and Jul 2). A dedicated
all-zero scan then pinned it exactly:

```
eastward_wind_at_ten_meters:   [728:736]  2023-04-02 00:00 .. 21:00  (8 frames, all zero)
northward_wind_at_ten_meters:  [728:736]  ... same
PRMSL:                         [728:736]  ... same
PRATEsfc:                      [728:736]  ... same
air_temperature_at_two_meters: [728:736]  ... same
UNION: 8 steps, 2023-04-02 only.
```

Rank 0's batch 92 did NOT complete in the ~68 min before the SIGABRT.

## Fix applied — DONE 2026-09-08

1. `video_inference.yaml` (batch_size 1, narrow subset, max_batches 1) —
   regenerated only the Apr 2 clip into a scratch store, single-GPU
   (job `01M1ZSE66PABVDNK7608QM3TS5`, ~76 min).
2. `merge_apr02_clip.py` — per-channel idempotent: writes a channel if its
   Apr 2 region is the gap (all-zero *or* NaN), skips it if it already
   matches the scratch clip, aborts only on a real mismatch; PRATE-safe
   emptiness check (precip is legitimately >90 % exact zeros); final
   whole-store scan over all 5 channels. Ran in two passes
   (`01M1ZY2A7...` wrote u10/v10/PRMSL then aborted on the old PRATE check;
   `01M1ZY9WX...` finished PRATEsfc + T2m). Final scan: **all 5 channels,
   no empty frames anywhere** — store complete.
3. Re-ran `crps_eval` (`01M1ZYRSMMT8MYSFKWEQNR2K1S`) — corrected two-block
   numbers now sane (PRMSL MSE 0.43 mb², was 81677). `diurnal_cycle_eval`
   was unaffected (JJA doesn't touch Apr 2). Results in
   `toy/two_block_vs_joint_gen.md`.
4. Scratch `RESUME-2023-04-02-clip.zarr` deleted from weka.

The TC / mid-latitude-cyclone analysis (`two_block_family_analysis.py`) was
**not** affected — tracks 789 (May-Jun), 795 (Jun-Jul), 829 (Nov) and the
Jan ETCs don't touch Apr 2.

## Two real bugs worth fixing upstream

- **ZarrWriter `fill_value=0.0`** — should be NaN, so a partial/failed write
  is detectable (0.0 is a legitimate value for wind and a plausible-looking
  one for a normalized field; NaN is unambiguous).
- **The trailing `dist.barrier()`** in `video_inference.run_inference` only
  gates the "Completed inference" log line (ranks write disjoint time
  regions). A longer process-group timeout, or dropping that barrier, would
  keep a near-complete multi-day run from exiting 1 with a silent 1-clip gap.
