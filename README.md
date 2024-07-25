# ACE: AI2 Climate Emulator
This repo contains the inference code accompanying "ACE: A fast, skillful learned global atmospheric model for climate prediction" ([arxiv:2310.02074](https://arxiv.org/abs/2310.02074)).

## DISCLAIMER
This is rapidly changing research software. We make no guarantees of maintaining backwards compatibility.

## Quickstart

### 1. Install

Clone this repository. Then assuming [conda](https://docs.conda.io/en/latest/)
is available, run
```
make create_environment
```
to create a conda environment called `fme` with dependencies and source
code installed. Alternatively, a Docker image can be built with `make build_docker_image`.
You may verify installation by running `pytest fme/`.

### 2. Download data and checkpoint

The checkpoint and a 1-year subsample of the validation data are available at
[this Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.10791086).
Download these to your local filesystem.

Alternatively, if interested in the complete dataset, this is available via a public
[requester pays](https://cloud.google.com/storage/docs/requester-pays)
Google Cloud Storage bucket. For example, the 10-year validation data (approx. 190GB)
can be downloaded with:
```
gsutil -m -u YOUR_GCP_PROJECT cp -r gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs/validation .
```
It is possible to download a portion of the dataset only, but it is necessary to have
enough data to span the desired prediction period. The checkpoint is also available on GCS at
`gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/checkpoints/ace_ckpt.tar`.

### 3. Update configuration and run
Update the paths in the [example config](examples/config-inference.yaml). Then in the
`fme` conda environment, run inference with:
```
python -m fme.fcn_training.inference.inference examples/config-inference.yaml
```

## Configuration options
See the `InferenceConfig` class in [this file](fme/fme/fcn_training/inference/inference.py) for
description of configuration options. The [example config](examples/config-inference.yaml)
shows some useful defaults for performing a 400-step simulation (100 days, with the 6-hour time step).

## Performance
While inference can be performed without a GPU, it may be very slow in that case. In addition,
I/O performance is critical for fast inference due to loading of forcing data and target data
during inference.

## Analyzing output
Various climate performance metrics are computed online by the inference code. These can be viewed via
[wandb](https://wandb.ai) by setting `logging.log_to_wandb` to true and updating `logging.entity`
to your wandb entity. Additionally, raw output data is saved to netCDF by the inference code.

## Available datasets
Two versions of the dataset described in [arxiv:2310.02074](https://arxiv.org/abs/2310.02074)
are available:
```
gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-zarrs
gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs
```
The `zarr` format is convenient for ad-hoc analysis. The netCDF version contains our
train/validation split which was used for training and inference.
