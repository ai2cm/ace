# Inter-model dispersion summary

Per-variable dispersion of std, d1_std, and mean across 22 models (one value per model, averaged over realizations and experiments).

**Flags**: CoV > 30% on std/d1_std, or model-mean spread > 0.5σ of within-model variability — these are the variables where shared scales lose meaningful conditioning vs per-dataset.

## std (cell-time)

| variable | n | median | max/median | CoV |
|---|---:|---:|---:|---:|
| hus (plev_index=7) | 19 | 2.88e-07 | 2.31 | 50% |
| hus (plev_index=5) | 19 | 5.9e-07 | 2.47 | 45% |
| hus (plev_index=6) | 19 | 3.14e-07 | 2.51 | 36% |

## d1_std (one-step finite difference)

| variable | n | median | max/median | CoV |
|---|---:|---:|---:|---:|
| hus (plev_index=5) | 19 | 2.64e-07 | 2.91 | 55% |
| hus (plev_index=7) | 19 | 5.05e-08 | 2.77 | 52% |
| hus (plev_index=6) | 19 | 8.21e-08 | 2.51 | 41% |

## mean (in std-units)

| variable | n | median mean | spread (σ) | range (σ) |
|---|---:|---:|---:|---:|
| hus (plev_index=6) | 19 | 2.47e-06 | 1.59 | 7.03 |
| hus (plev_index=7) | 19 | 2.91e-06 | 1.32 | 4.95 |
| hus (plev_index=5) | 19 | 2.62e-06 | 1.18 | 5.09 |
| ta_derived_layer_6 | 19 | 219 | 0.51 | 1.84 |

## All variables — std dispersion (full table)

| variable | n | median std | max/median | CoV |
|---|---:|---:|---:|---:|
| huss | 20 | 0.00596 | 1.28 | 11% |
| pr | 19 | 6.27e-05 | 1.12 | 6% |
| psl | 20 | 1.14e+03 | 1.12 | 11% |
| tas | 20 | 15.3 | 1.04 | 12% |
| hus (plev_index=0) | 19 | 0.00569 | 1.08 | 6% |
| hus (plev_index=1) | 19 | 0.00411 | 1.07 | 6% |
| hus (plev_index=2) | 19 | 0.00269 | 1.11 | 7% |
| hus (plev_index=3) | 19 | 0.00127 | 1.18 | 12% |
| hus (plev_index=4) | 19 | 8.79e-05 | 1.31 | 24% |
| hus (plev_index=5) | 19 | 5.9e-07 | 2.47 | 45% |
| hus (plev_index=6) | 19 | 3.14e-07 | 2.51 | 36% |
| hus (plev_index=7) | 19 | 2.88e-07 | 2.31 | 50% |
| ua (plev_index=0) | 19 | 7.14 | 1.15 | 7% |
| ua (plev_index=1) | 19 | 8.06 | 1.07 | 5% |
| ua (plev_index=2) | 19 | 8.87 | 1.06 | 3% |
| ua (plev_index=3) | 19 | 11.5 | 1.08 | 3% |
| ua (plev_index=4) | 19 | 18 | 1.06 | 4% |
| ua (plev_index=5) | 19 | 15.4 | 1.10 | 9% |
| ua (plev_index=6) | 19 | 16.4 | 1.09 | 9% |
| ua (plev_index=7) | 19 | 24.8 | 1.22 | 9% |
| va (plev_index=0) | 19 | 5.38 | 1.06 | 4% |
| va (plev_index=1) | 19 | 5.22 | 1.10 | 5% |
| va (plev_index=2) | 19 | 5.54 | 1.14 | 5% |
| va (plev_index=3) | 19 | 7.29 | 1.13 | 6% |
| va (plev_index=4) | 19 | 11.6 | 1.10 | 8% |
| va (plev_index=5) | 19 | 6.95 | 1.07 | 6% |
| va (plev_index=6) | 19 | 5.5 | 1.05 | 6% |
| va (plev_index=7) | 19 | 8.02 | 1.16 | 8% |
| zg (plev_index=0) | 19 | 921 | 1.14 | 5% |
| zg (plev_index=1) | 19 | 555 | 1.19 | 8% |
| zg (plev_index=2) | 19 | 343 | 1.14 | 7% |
| zg (plev_index=3) | 19 | 280 | 1.07 | 5% |
| zg (plev_index=4) | 19 | 482 | 1.08 | 5% |
| zg (plev_index=5) | 19 | 471 | 1.14 | 10% |
| zg (plev_index=6) | 19 | 443 | 1.13 | 11% |
| zg (plev_index=7) | 19 | 686 | 1.17 | 8% |
| ta_derived_layer_0 | 19 | 14.8 | 1.22 | 8% |
| ta_derived_layer_1 | 19 | 13.1 | 1.13 | 5% |
| ta_derived_layer_2 | 19 | 12.1 | 1.06 | 5% |
| ta_derived_layer_3 | 19 | 10.2 | 1.11 | 6% |
| ta_derived_layer_4 | 19 | 5.33 | 1.17 | 7% |
| ta_derived_layer_5 | 19 | 9.73 | 1.07 | 5% |
| ta_derived_layer_6 | 19 | 6.78 | 1.27 | 10% |
| hfls | 19 | 70.5 | 1.07 | 3% |
| hfss | 18 | 29.4 | 1.11 | 6% |
| rlds | 18 | 73.5 | 1.05 | 4% |
| rlus | 18 | 75.7 | 1.04 | 3% |
| rlut | 17 | 42.6 | 1.04 | 4% |
| rsds | 19 | 95.2 | 1.03 | 1% |
| rsus | 17 | 37.3 | 1.08 | 4% |
| siconc | 10 | 17 | 1.09 | 16% |
| ts | 22 | 15.7 | 1.04 | 3% |
| orog | 21 | 593 | 1.06 | 2% |
| sftlf | 22 | 42.2 | 1.04 | 2% |
| sfcWind | 19 | 3.49 | 1.08 | 4% |
| uas | 15 | 5.31 | 1.07 | 3% |
| vas | 15 | 4.25 | 1.11 | 5% |
