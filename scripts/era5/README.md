# Scripts for generating a dataset for Full Model Emulation based on ERA5

## Downloading some 2D variables from NCAR mirror of ERA5

Most of this dataset will be generated from Google's version of the ERA5
dataset (see https://github.com/google-research/arco-era5). However, that dataset
is missing some variables such as sensible and latent heat flux and some radiative
fluxes. Therefore, we download these variables, as well as some auxiliary variables
such as land-fraction and surface geopotential, from NCAR's hosted version of ERA5.
NCAR has the 0.25° regular lat-lon data, so we download that version. Regridding and
conversion to zarr is left to a future step.

This download step should only have to be performed once. It can be started via
```
make ingest_ncar_variables
```
and uses an argo workflow to run on our Google cloud resources. Sometimes the download
will fail for certain variables. If this happens, the workflow can be resubmitted
with the same command as above, and it will pick up where it left off.

## Converting netCDF files downloaded from NCAR to zarr

To facilitate further processing and alignment with the data available from
Google, we use an xarray-beam pipeline to concatenate, merge, and rechunk the
ERA5 data downloaded from NCAR into a set of three zarr stores:

- `e5.oper.fc.sfc.meanflux`
- `e5.oper.an.sfc`
- `e5.oper.invariant`

The scaled up version of the beam pipeline is run using Dataflow. It first
requires creating a local Python environment with the needed dependencies
installed:

```
make create_environment
```

To submit the full Dataflow workflow, one can use:

```
make netcdf_to_zarr_dataflow
```

This submits the jobs to create each dataset one at a time, though the process
for creating each dataset is highly parallelized.

If needed the Docker image required for running the workflow in the cloud can
be rebuilt and pushed using:

```
make build_dataflow push_dataflow
```

## Computing coarsened ERA5 dataset for FME

Once the previous steps have been done, all the necessary data should be available
in zarr format on Google Cloud Storage. Now it is possible to compute all necessary
variables on the 1° horizontal resolution and with eight vertical layers. This is
done using an xarray-beam pipeline similar to the previous step.

First, if not already available, build a docker image using the same instructions
as in previous step. Additionally, create the local "era5-ingestion" conda
environment.

Once these steps are done, the workflow can be submitted with

```
make era5_dataflow
```
