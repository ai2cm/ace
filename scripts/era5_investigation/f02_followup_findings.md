# ERA5 Follow-up Analysis: Discontinuities and Radiation Negatives

## Context

This follow-up investigates two questions raised during PR review of the initial findings:

1. Are the 1986 and 1993 discontinuities (Finding 4) real volcanic signals or reanalysis artifacts?
2. Why does USWRFsfc have many small negative values while DSWRFsfc and USWRFtoa do not? (Finding 5)

Analysis scripts: `04_volcanic_vs_artifact.py`, `05_uswrf_negatives.py`, `06_plot_discontinuities.py`. Plots in `plots/`.

---

## Finding 4 Revisited: Volcanic Events vs Reanalysis Artifacts

**The original Finding 4 conflated two different phenomena. Some jumps are reanalysis artifacts, and some are real volcanic signals.**

### 1986 specific_total_water_0 (-3.9σ): REANALYSIS ARTIFACT

Weekly global-mean data shows the drop happens in a single week — March 30 to April 6, 1986 — a sharp ~10% decrease perfectly aligned with the **April 1, 1986 ERA5 parallel stream merge**.

| Week ending | specific_total_water_0 (global mean) |
| ----------- | ------------------------------------ |
| 1986-03-23  | 3.178e-06                            |
| 1986-03-30  | 3.159e-06                            |
| 1986-04-06  | 2.899e-06 ← sharp drop              |
| 1986-04-13  | 2.859e-06                            |

A volcanic signal from El Chichón (erupted March 1982) would have appeared 1982–1984 and decayed gradually. A sharp drop 4 years later at an exact stream merge date is not volcanic. The year-over-year analysis in the initial finding made this look like a January boundary jump only because it compared annual means.

### 1993 air_temperature_0 (-3.1σ) and air_temperature_1 (-2.5σ): VOLCANIC (Pinatubo)

Mt. Pinatubo erupted June 15, 1991. The data shows:

- Clear stratospheric **warming** of ~0.9K in 1991–1992 (the expected sulfate aerosol heating signature)
- Cooling in 1993 as aerosol dissipated
- Smooth transition at the Sep 1992 ERA5 stream boundary (no sharp discontinuity in weekly data)

| Year | air_temperature_0 | Change from prior year |
| ---- | ------------------ | ---------------------- |
| 1990 | 222.13 K           | (pre-eruption baseline)|
| 1991 | 222.90 K           | +0.77 K (warming)      |
| 1992 | 223.06 K           | +0.17 K (peak)         |
| 1993 | 222.11 K           | -0.95 K (decay)        |
| 1994 | 221.88 K           | -0.23 K (recovery)     |

This warming→cooling pattern matches Pinatubo literature exactly. The ERA5 Stream 2→3 boundary (Sep 1992) may contribute a minor superimposed artifact, but the dominant signal is real volcanic physics.

### 2000 specific_total_water_0 (-2.5σ): LIKELY ARTIFACT

This aligns with a known Jan 2000 ERA5 stream boundary. The spatial before/after plot shows a near-uniform global negative offset — characteristic of a systematic bias shift, not weather or volcanic signals.

### Summary of Discontinuity Classification

| Year | Variable               | σ    | Classification      | Evidence                                     |
| ---- | ---------------------- | ---- | ------------------- | -------------------------------------------- |
| 1986 | specific_total_water_0 | -3.9 | Reanalysis artifact | Sharp drop at Apr 1 stream merge date        |
| 1993 | air_temperature_0      | -3.1 | Volcanic (Pinatubo) | Matches eruption warming→cooling signature   |
| 1993 | air_temperature_1      | -2.5 | Volcanic (Pinatubo) | Same as above                                |
| 2000 | specific_total_water_0 | -2.5 | Reanalysis artifact | Known stream boundary; uniform spatial offset |

### Should volcanic aerosol be a forcing?

The Pinatubo signal is real and large (~0.9K stratospheric warming). The training data includes multiple major volcanic events:

- Mt. Agung (1963)
- El Chichón (1982)
- Mt. Pinatubo (1991)

Without a volcanic aerosol forcing variable (e.g., stratospheric aerosol optical depth), the model has no way to predict these events — they just add noise to the loss. Two options:

1. **Add stratospheric aerosol optical depth as a forcing** — would let the model learn volcanic responses, important if ACE should handle volcanic scenarios
2. **Exclude major volcanic periods from training** — simpler, but loses data and limits applicability

---

## Finding 5 Revisited: USWRFsfc Negative Values are Floating-Point Noise

**The 23% negative values in USWRFsfc are not physically meaningful — they are floating-point noise from ERA5 post-processing.**

### Evidence

- The negative values are negligibly small: **median = -0.00002 W/m²**, max magnitude = 0.06 W/m²
- Only **5 values out of 95 million** exceed -0.01 W/m² in absolute value
- **99.3% of negatives occur at nighttime** (where DSWRFsfc = 0)
- They are uniformly distributed spatially (slightly more at poles due to longer nights)

| Threshold    | Count exceeding | % of total |
| ------------ | --------------- | ---------- |
| < 0          | 21,897,838      | 23.08%     |
| < -0.01      | 5               | 0.000005%  |
| < -0.1       | 0               | 0%         |

### Why USWRFsfc but not DSWRFsfc or USWRFtoa?

DSWRFsfc and USWRFtoa use exact zeros for nighttime (29% exact zeros each). USWRFsfc went through a different interpolation or processing step in the ERA5 pipeline that introduced tiny floating-point residuals (~10⁻⁵ W/m²) instead of hard-clamping to zero. USWRFsfc has only 0.8% exact zeros — the remaining nighttime values are these tiny negatives.

### Implication for training

These are negligible — 6 orders of magnitude below the signal (mean USWRFsfc = 35 W/m²). They should not affect model training. The "23% negative" statistic from the initial finding is technically correct but misleading — it really means "23% are nighttime values that should be exactly zero but have ~10⁻⁵ W/m² floating-point noise."
