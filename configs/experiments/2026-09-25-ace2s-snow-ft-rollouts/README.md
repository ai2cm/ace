# ACE2S snow treatment: holdout rollouts of the fine-tuned checkpoints, 1-deg daily

Offline evaluation of the two fine-tuned treatment models at their best-inference epochs, the
nominal final products of the `2026-09-17-ace2s-snow-inline-metrics-train` round, over each
dataset's holdout period with overlapping members to sample ensemble variability. The inline
five-year rollout during training rests on a few initial conditions; these runs confirm its
time-mean climate, memory and snow-season statistics on independent dates and save daily fields
for offline analysis.

## Arms

| arm | fine-tune result dataset | best-inference epoch | inline time-mean channel-mean RMSE |
|---|---|---|---|
| cm4-masked-naive | `01M38TTH492G0WT71YFAEH0V58` | 13 of 21 run (the run destabilised after epoch 14; see the training README) | 0.016 |
| era5-masked-naive | `01M37GMA67V8201AN8NVASX7ZP` | best of 40 | 0.037-0.040 |

## Initial conditions

Eight members per arm, starting 1 January, overlapping so that they sample the spread at
matched dates while covering as much of each holdout as it allows:

- **cm4**: ten-year members (3650 steps) starting every four years, model years 0311, 0315,
  ..., 0339, so consecutive members overlap by six years and together cover 0311-0349. Training
  stops at 0306 and validation covers 0306-0311, so every member and its forcing lie in the
  untouched 0311-0351 tail of the piControl run. Ten years is twice the inline horizon, which
  also tests the checkpoint's stability beyond the range where the fine-tune later drifted.
- **era5**: five-year members (1825 steps) starting 1998-2005, overlapping by four years, all
  inside the 1998-2010 holdout (training used 1995 and earlier, 2011-2019 and 2021 onward;
  validation 1996-1997), so forcing never comes from a training year. The inline configs'
  members all start in 1998-2000.

## Output

`gs://vcm-ml-intermediate/2026-09-25-ace2s-snow-ft-rollouts/<arm>/`: the standard evaluator
diagnostics, the `anomaly_memory` and `snow_season` aggregators on the training configs' boxes,
and daily fields (the 2026-09-13 snow-memory set plus `PRATEsfc`) as zarr, cropped to
`lat_extent [-40, 90]`, with the target written alongside.

## Launch

```
cd configs/experiments/2026-09-25-ace2s-snow-ft-rollouts
./run-ace-evaluator.sh          # both arms; W&B group ace2s-snow-inline-metrics, like the training jobs
./run-ace-evaluator.sh cm4      # one arm
```
