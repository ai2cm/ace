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

### Generating AIMIP evaluation initial conditions

`create_aimip_ic_datasets.py` builds the ICs a model is initialized from for AIMIP inference.
Each member is one timestamp from the ERA5 zarr, restamped to a common target time.

```make create_aimip_ics```

- `ERA5_GCS_DATA` — source zarr; use the build the model was trained on.
- `GCS_IC_PATH` — upload destination.
- `IC_EXTRA_ARGS` — e.g. `--include-near-surface` for models carrying
  `TMP2m`/`Q2m`/`UGRD10m`/`VGRD10m`. `--target-timestamp` and `--ic-timestamp` set the
  restamped and source times.

ICs carry only prognostic variables; forcings come from the forcing zarr at inference time.

### Regenerating the forcing for a different ERA5 build

`HGTsfc`, `DSWRFtoa` and `land_fraction` are read from an ACE ERA5 zarr, and differ between
builds (the March 2026 pipeline rewrite moves `HGTsfc` by tens of metres in mountains), so
that zarr should match the model's training build:

```
FORCING_EXTRA_ARGS="--ace2-era5-gcs-data gs://vcm-ml-intermediate/<era5>.zarr" \
  make process_aimip_forcing
```

The source must span the whole window or the run errors; `--extension-start` opts into
synthesizing the remainder, needed only for short sources like the 2022-ending
`2024-06-20-…` store.

### Generating the public AIMIP forcing dataset

Additionally, the public forcing dataset at 0.25° resolution [available on Zenodo as version 2](https://zenodo.org/records/17065758) can also be generated and uploaded to GCS here. To do so, run:

```make create_aimip_forcing```
