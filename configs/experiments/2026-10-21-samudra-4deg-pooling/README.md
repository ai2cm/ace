# Pooling strategies for Samudra at 4 degrees (2026-10-21)

Samudra pools by 2 at each of its four U-Net levels, which 45x90 divides only
once, so every deeper level has an odd height or width. This experiment trains
one seed of four ways to handle that, as compared in the ace#1589 pre-review
report ([ai2cm/reports#183](https://github.com/ai2cm/reports/pull/183)):

| arm | builder option | 4 degree bottleneck |
|---|---|---|
| `_floor` | none: x2 floor pooling, the decoder refills dropped rows and columns | 2x5 |
| `_padpool` | `pad_pool: true` ([ace#1589](https://github.com/ai2cm/ace/pull/1589)) | 3x6 |
| `_conspool` | `conservative_pool: true`: area-exact pooling to `ceil(n / 2)`, linear + correction upsampling | 3x6 |
| `_exact3311` | `pooling: [3, 3, 1, 1]` ([ace#1590](https://github.com/ai2cm/ace/pull/1590)) | 5x10 |

All four share `ocean_socorr_4deg/train-config.yaml`: the 4 degree twin of the
1 degree run `01M5FC4FVVQC2JFAB94XBMXJQ2` (James Duncan's `ocean_socorr` with
the SSH salt correction, the input mask filled with the mean, an uncapped
GELU and `lat_pad: pole`), at dilation `[1, 2, 4, 2]`. That dilation keeps
every level's pole pad within its height (the floor bottleneck's 2 rows, the
#1590 arm's 5 rows under a dilation of 4), so no far-pole continuation is
needed, and it is left out of this branch.

## Code on this branch

`main`, plus `origin/job_runner`, ace#1589 (`feature/samudra-pool-padding`)
and ace#1590 (`feature/samudra-levpool`) merged, `gelu_cap` (from
`exp/2026-10-09-samudra-pool-padding`) and the conservative pool. Under
explicit `pooling`, `pad_pool` and `conservative_pool` are rejected.

## Running

```bash
job_runner/uncoupled_train.sh configs/experiments/2026-10-21-samudra-4deg-pooling ocean_socorr_4deg \
  --coupled_stats elynn/2026-07-22-cm4-1pctco2-4deg-coupled-stats
```
