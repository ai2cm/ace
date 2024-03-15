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