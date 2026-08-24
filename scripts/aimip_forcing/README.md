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

`create_aimip_ic_datasets.py` builds the initial-condition files an ACE model is
initialized from for AIMIP inference. Each IC member is a single timestamp selected from
the 1-degree 8-layer ERA5 zarr and restamped to a common target time, so the members are
an ensemble over initial state rather than over start date.

```make create_aimip_ics```

Relevant variables:

- `ERA5_GCS_DATA` — source zarr. Must be the same ERA5 version the model was trained on.
- `GCS_IC_PATH` — where the resulting `.nc` files are uploaded.
- `IC_EXTRA_ARGS` — extra flags. Use `--include-near-surface` for models that carry
  `TMP2m`/`Q2m`/`UGRD10m`/`VGRD10m` as prognostic inputs; the source zarr must have them.
  `--target-timestamp` and `--ic-timestamp` set the restamped time and the source times,
  and must exist in the source zarr.

Note the ICs supply only *prognostic* variables. Forcings (`DSWRFtoa`, `HGTsfc`,
`land_fraction`, `ocean_fraction`, `sea_ice_fraction`, `surface_temperature`) come from the
forcing zarr at inference time, so they are deliberately absent here.

### Generating the public AIMIP forcing dataset

Additionally, the public forcing dataset at 0.25° resolution [available on Zenodo as version 2](https://zenodo.org/records/17065758) can also be generated and uploaded to GCS here. To do so, run:

```make create_aimip_forcing```
