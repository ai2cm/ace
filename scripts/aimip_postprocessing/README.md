# AIMIP Postprocessing

Post-process ACE AIMIP inference results into CMIP6-compliant netCDF files and upload them to GCS and DKRZ object storage.

The script reads raw output netCDFs from GCS, applies CF-convention transformations (coordinate standardization, vertical dimension stacking, bounds computation, metadata), and writes the results locally before uploading.

## Contents

| File | Description |
|---|---|
| `postprocess.py` | Main processing script |
| `simulations.yaml` | List of simulations to process (`SimulationConfig`) |
| `files.yaml` | List of variable/table/grid combinations to process (`FileConfig`) |
| `test_postprocess.py` | Unit tests for key helper functions |
| `Makefile` | Convenience targets |

## Requirements

The `fme` conda environment has the required dependencies. If you are running outside of that environment, install:

```
fsspec gcsfs s3fs xarray cftime python-dotenv click pyyaml netcdf4
```

## Environment variables

DKRZ upload requires `ACE_KEY` and `ACE_SECRET`. Place them in a `.env` file in the working directory:

```
ACE_KEY=<your-key>
ACE_SECRET=<your-secret>
```

## Usage

Process all simulations (uploads to GCS and DKRZ):

```bash
python postprocess.py
```

Process a single simulation, skipping both uploads:

```bash
python postprocess.py \
    --simulation ace-aimip-inference-oct-1978-2024-IC1 \
    --skip-gcs-upload \
    --skip-dkrz-upload
```

Skip only DKRZ:

```bash
python postprocess.py --skip-dkrz-upload
```

Use custom simulation/file lists:

```bash
python postprocess.py \
    --simulations-file my_simulations.yaml \
    --files-file my_files.yaml
```

Override source/destination paths:

```bash
python postprocess.py \
    --raw-results-dir gs://my-bucket/raw \
    --processed-results-dir gs://my-bucket/processed \
    --output-version v20260101
```

Run `python postprocess.py --help` for the full option list.

## Output layout

Local (temporary) and GCS outputs follow this directory structure:

```
{local_dir}/{experiment_id}/{variant_label}/{table_id}/{varname}/{grid_label}/{output_version}/{filename}.nc
```

For example:

```
/tmp/aimip-ace/aimip/r1i1p1f1/Amon/tas/gn/v20251130/tas_Amon_ACE2-ERA5_aimip_r1i1p1f1_gn_197810-202412.nc
```

DKRZ outputs are uploaded to `ai-mip/Ai2/ACE2-ERA5/` with the same relative structure.

## Running tests

```bash
python -m pytest test_postprocess.py -v --noconftest
```
