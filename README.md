# ACE: AI2 Climate Emulator
This repo constains the inference code accompanying "ACE: A fast, skillful learned global atmospheric model for climate prediction" ([arxiv:2310.02074](https://arxiv.org/abs/2310.02074)).

## DISCLAIMER
This is rapidly changing research software. We make no guarantees of maintaining backwards compatibility.

## Quickstart

### 1. Clone this repository and install dependencies

Assuming [conda](https://docs.conda.io/en/latest/) is available, run
```
make create_environment
```
to create a conda environment called `fme` with dependencies and source
code installed. Alternatively, a Docker image can be built with `make build_docker_image`.
You may verify installation by running `pytest fme/`.

### 2. Download data and checkpoint

These are available via a public
[requester pays](https://cloud.google.com/storage/docs/requester-pays)
Google Cloud Storage bucket. The checkpoint can be downloaded with:
```
gsutil -u YOUR_GCP_PROJECT cp gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/checkpoints/ace_ckpt.tar .
```
Download the 10-year validation data (approx. 190GB; can also download a portion only,
but it is required to download enough data to span the desired prediction period):
```
gsutil -m -u YOUR_GCP_PROJECT cp -r gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs/validation .
```

### 3. Update the paths in the [example config](examples/config-inference.yaml).
Then in the `fme` conda environment, run inference with:
```
python -m fme.fcn_training.inference.inference examples/config-inference.yaml
```

## Configuration options
See the `InferenceConfig` class in [this file](fme/fme/fcn_training/inference/inference.py) for
description of configuration options. The [example config](examples/config-inference.yaml)
shows some useful defaults for performing a 400-step (100-day) simulation (e.g. using a 6-hr time-step)

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
