# Data processing for full model emulation training

This directory contains scripts for generating various datasets needed for FME training, including the FV3GFS primary, baseline, and stats datasets.

It also contains scripts for generating E3SM training data.

The first step in the process to create intermediate datasets (e.g. `make fv3gfs_AMIP_dataset`) uses argo, and can be run on your Google VM.
See the vcm-workflow-control repo for instructions on how to install and run argo.

The second step, which produces monthly netCDF files locally (e.g. `make fv3gfs_AMIP_monthly_netcdfs`), can be run on cirrascale in an interactive session.
To create an interactive session, run the following command from the `scripts/data_process` directory:

```
beaker session create --budget ai2/climate --image beaker://jeremym/fme-2bc0033e --gpus 0 --mount hostPath:///net/nfs/climate=/net/nfs/climate --mount hostpath://$(pwd)=/full-model --workdir /full-model/scripts/data_process
```

Doing so will require that your current working directory is a mountable path (e.g. something in /data).
If you'd like to write to a different directory than /net/nfs/climate, you can mount that path instead.

Once inside the image, you will need to authorize access to GCS by running `gcloud auth application-default login` and following the instructions, including to run `gcloud config set project vcm-ml` afterwards.

You can then produce the monthly netCDFs in a target directory by modifying the `OUTPUT_DIR` or `OUTPUT_DIR_AMIP` variable in the make command below.

```
make fv3gfs_AMIP_monthly_netcdfs RESOLUTION=4deg OUTPUT_DIR_AMIP=/data/shared/2023-12-20-vertically-resolved-4deg-fme-amip-ensemble-dataset
```

The stats dataset creation step (e.g. `make fv3gfs_AMIP_stats_beaker_dataset`) must be run in the fme conda environment (created by `make create_environment` at the top level of this repo), and additionally requires the beaker client is installed ([install instructions](https://beaker-docs.apps.allenai.org/start/install.html)).
