### AIMIP Forcing Scripts

This directory provides scripts to generate a forcing dataset for use with an ACE model checkpoint during inference. The scripts process public AIMIP data available at [Zenodo (AIMIP Forcing Data)](https://zenodo.org/records/16782373).

#### Contents

- **Data Download:** Fetch raw AIMIP forcing datasets from Zenodo.
- **Regridding:** Regrid the quarter-degree AIMIP forcing data to the 1 degree Gaussian grid used by ACE.
- **Time resampling and dataset generation:** After regridding, interpolate the monthly-mean AIMIP forcing data to the 6-hourly resolution used by ACE.
- **Upload:** Upload the resulting zarr dataset to GCS.

#### Environment and workflow

To install the required `xesmf` package to run the workflow, a Conda environment called `regrid-aimip-forcing` can be created with:

```make create_environment```

Once the environment is installed and active, run the full workflow (regrdding the AIMIP forcing data, performing time resampling and uploading) with:

```make process_aimip_forcing```

The GCS path for the resulting zarr dataset can be specified:

```GCS_OUTPUT_PATH=gs://vcm-ml-intermediate/2025-08-27-era5-1deg-monthly-mean-forcing-1978-2024.zarr make process_aimip_forcing```

Note that the workflow is memory-intensive and was run on a high-memory (128GB) GCP VM.

### Daily variant (daily 1° model)

The submitted model steps six-hourly; the daily 1° model (e.g. `nobgd4ek`) steps
once per day and additionally carries near-surface prognostics (`TMP2m`, `Q2m`,
`UGRD10m`, `VGRD10m`). Two things therefore differ from the six-hourly workflow:
the forcing is resampled to a daily step, and the initial conditions must include
those near-surface fields. The spatial regrid is shared (identical 1° grid).

Build the daily forcing zarr first, then the evaluation datasets
(repeated-first-step forcing + ICs), which read the forcing zarr back from GCS:

```
make process_daily_forcing                  # build + upload the daily forcing zarr
make create_daily_aimip_evaluation_datasets # prepend first step + build ICs
```

`process_daily_forcing` interpolates the shared regridded monthly forcing onto
the daily zarr's own time coordinate and sources insolation and `HGTsfc` from
that zarr (`interpolate_aimip_forcing.py --extension-start ""`; because the daily
zarr already spans the full window, no synthetic extension or insolation
repeating is used). `create_daily_aimip_evaluation_datasets` then builds ICs with
`--include-near-surface`.

Confirm before running:

- `DAILY_ERA5_GCS_DATA` — the GCS path of the daily zarr the model trained on
  (mounted in training as
  `/climate-default/2026-03-19-era5-1deg-8layer-daily-1940-2025.zarr`).
- The daily calendar hour is assumed to be 06Z. The timestamps *selected from*
  the daily zarr must fall on that hour: the IC source members
  (`DAILY_IC_TIMESTAMPS`) and the repeated first forcing step
  (`DAILY_FORCING_FIRST_STEP`). The assigned IC/prepend target
  (`DAILY_IC_TARGET`) is only a relabeling and need not exist in the zarr.

### Generating the public AIMIP forcing dataset

Additionally, the public forcing dataset at 0.25° resolution [available on Zenodo as version 2](https://zenodo.org/records/17065758) can also be generated and uploaded to GCS here. To do so, run:

```make create_aimip_forcing```
