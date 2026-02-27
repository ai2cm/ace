# compute_enso_index

Computes the ENSO/Niño 3.4 index from SST forcing data.

## Environment

Activate the repo's default conda environment before running:

```bash
conda activate fme
```

## Usage

Run via the provided Makefile targets from this directory:

```bash
make amip_enso_index        # run with default AMIP SST data
make era5_aimip_enso_index  # run with ERA5 AIMIP data
```
