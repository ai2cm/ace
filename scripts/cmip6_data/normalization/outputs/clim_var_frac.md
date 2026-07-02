# Climatology variance fraction across models

Per-variable share of total variance explained by the seasonal + spatial climatology, with dispersion across models. **clim_var_frac near 1** = mostly explained by the time-mean field (removing climatology helps a lot). **near 0** = anomalies dominate (climatology removal is nearly a no-op).

Sorted by range across models — large range = the climatology/anomaly split *itself* is model-dependent.

| variable | n | median | IQR | min | max | range |
|---|---:|---:|---:|---:|---:|---:|
| hus (plev_index=7) | 19 | 0.72 | 0.28 | 0.10 | 0.83 | 0.72 |
| hus (plev_index=6) | 19 | 0.37 | 0.23 | 0.20 | 0.87 | 0.67 |
| hus (plev_index=5) | 19 | 0.23 | 0.17 | 0.15 | 0.53 | 0.39 |
| ta_derived_layer_4 | 19 | 0.37 | 0.09 | 0.29 | 0.64 | 0.35 |
| ua (plev_index=5) | 19 | 0.50 | 0.07 | 0.33 | 0.65 | 0.32 |
| psl | 20 | 0.54 | 0.06 | 0.31 | 0.62 | 0.31 |
| zg (plev_index=6) | 19 | 0.55 | 0.06 | 0.35 | 0.65 | 0.30 |
| ua (plev_index=6) | 19 | 0.60 | 0.08 | 0.41 | 0.69 | 0.28 |
| zg (plev_index=7) | 19 | 0.32 | 0.06 | 0.21 | 0.45 | 0.23 |
| ua (plev_index=7) | 19 | 0.40 | 0.16 | 0.30 | 0.53 | 0.23 |
| tas | 20 | 0.85 | 0.02 | 0.68 | 0.87 | 0.19 |
| hus (plev_index=4) | 19 | 0.49 | 0.09 | 0.38 | 0.56 | 0.18 |
| hus (plev_index=3) | 19 | 0.42 | 0.05 | 0.33 | 0.51 | 0.18 |
| ta_derived_layer_5 | 19 | 0.67 | 0.04 | 0.63 | 0.80 | 0.17 |
| ua (plev_index=0) | 19 | 0.46 | 0.07 | 0.36 | 0.53 | 0.17 |
| huss | 20 | 0.85 | 0.02 | 0.74 | 0.90 | 0.16 |
| ua (plev_index=2) | 19 | 0.48 | 0.06 | 0.40 | 0.56 | 0.16 |
| ua (plev_index=1) | 19 | 0.49 | 0.06 | 0.40 | 0.56 | 0.16 |
| uas | 15 | 0.44 | 0.05 | 0.37 | 0.52 | 0.15 |
| ta_derived_layer_2 | 19 | 0.84 | 0.01 | 0.72 | 0.86 | 0.14 |
| hfss | 18 | 0.33 | 0.07 | 0.28 | 0.41 | 0.14 |
| ta_derived_layer_1 | 19 | 0.85 | 0.01 | 0.73 | 0.86 | 0.13 |
| ua (plev_index=4) | 19 | 0.40 | 0.05 | 0.35 | 0.48 | 0.13 |
| ta_derived_layer_0 | 19 | 0.87 | 0.01 | 0.75 | 0.88 | 0.13 |
| zg (plev_index=5) | 19 | 0.77 | 0.02 | 0.70 | 0.82 | 0.12 |
| rsds | 19 | 0.36 | 0.03 | 0.29 | 0.40 | 0.11 |
| ua (plev_index=3) | 19 | 0.43 | 0.05 | 0.39 | 0.49 | 0.10 |
| siconc | 10 | 0.82 | 0.05 | 0.76 | 0.87 | 0.10 |
| va (plev_index=7) | 19 | 0.13 | 0.04 | 0.08 | 0.18 | 0.10 |
| hfls | 19 | 0.53 | 0.03 | 0.51 | 0.61 | 0.10 |
| zg (plev_index=0) | 19 | 0.92 | 0.02 | 0.88 | 0.98 | 0.10 |
| vas | 15 | 0.18 | 0.03 | 0.14 | 0.23 | 0.09 |
| ta_derived_layer_6 | 19 | 0.11 | 0.02 | 0.08 | 0.17 | 0.09 |
| ta_derived_layer_3 | 19 | 0.84 | 0.01 | 0.77 | 0.87 | 0.09 |
| rlut | 17 | 0.52 | 0.04 | 0.49 | 0.58 | 0.09 |
| sfcWind | 19 | 0.51 | 0.01 | 0.49 | 0.58 | 0.09 |
| pr | 19 | 0.17 | 0.03 | 0.13 | 0.21 | 0.08 |
| hus (plev_index=2) | 19 | 0.53 | 0.02 | 0.50 | 0.58 | 0.08 |
| hus (plev_index=1) | 19 | 0.72 | 0.03 | 0.69 | 0.76 | 0.08 |
| va (plev_index=6) | 19 | 0.10 | 0.02 | 0.06 | 0.13 | 0.07 |
| zg (plev_index=3) | 19 | 0.84 | 0.02 | 0.80 | 0.86 | 0.06 |
| rsus | 17 | 0.43 | 0.02 | 0.40 | 0.46 | 0.06 |
| va (plev_index=0) | 19 | 0.15 | 0.01 | 0.12 | 0.17 | 0.05 |
| rlus | 18 | 0.86 | 0.01 | 0.84 | 0.89 | 0.04 |
| va (plev_index=5) | 19 | 0.08 | 0.01 | 0.04 | 0.09 | 0.04 |
| ts | 22 | 0.88 | 0.01 | 0.86 | 0.90 | 0.04 |
| rlds | 18 | 0.79 | 0.02 | 0.78 | 0.82 | 0.04 |
| va (plev_index=1) | 19 | 0.08 | 0.01 | 0.07 | 0.10 | 0.04 |
| hus (plev_index=0) | 19 | 0.85 | 0.01 | 0.84 | 0.87 | 0.04 |
| zg (plev_index=4) | 19 | 0.86 | 0.02 | 0.84 | 0.87 | 0.03 |
| zg (plev_index=2) | 19 | 0.93 | 0.02 | 0.91 | 0.94 | 0.03 |
| va (plev_index=2) | 19 | 0.06 | 0.01 | 0.05 | 0.08 | 0.03 |
| va (plev_index=4) | 19 | 0.04 | 0.01 | 0.03 | 0.06 | 0.03 |
| va (plev_index=3) | 19 | 0.04 | 0.01 | 0.03 | 0.05 | 0.02 |
| zg (plev_index=1) | 19 | 0.97 | 0.00 | 0.97 | 0.98 | 0.01 |
