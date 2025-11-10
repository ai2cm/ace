[![Docs](https://readthedocs.org/projects/ai2-climate-emulator/badge/?version=latest)](https://ai2-climate-emulator.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/fme.svg)](https://pypi.org/project/fme/)

<img src="ACE-logo.png" alt="Logo for the ACE Project" style="width: auto; height: 50px;">

# Ai2 Climate Emulator

Ai2 Climate Emulator (ACE) is a fast machine learning model that simulates global atmospheric variability in a changing climate over time scales ranging from hours to centuries.

This repo contains code accompanying five papers describing ACE models:
- "ACE: A fast, skillful learned global atmospheric model for climate prediction" ([link](https://arxiv.org/abs/2310.02074))
- "Application of the Ai2 Climate Emulator to E3SMv2's global atmosphere model, with a focus on precipitation fidelity" ([link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JH000136))
- "ACE2: Accurately learning subseasonal to decadal atmospheric variability and forced responses" ([link](https://www.nature.com/articles/s41612-025-01090-0))
- "ACE2-SOM: Coupling an ML Atmospheric Emulator to a Slab Ocean and Learning the Sensitivity of Climate to Changed CO<sub>2</sub>" ([link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JH000575))
- "Applying the ACE2 Emulator to SST Green's Functions for the E3SMv3 Global Atmosphere Model" ([link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000774))

## Installation

```
pip install fme
```

## Documentation

See complete documentation [here](https://ai2-climate-emulator.readthedocs.io/en/latest/) and a quickstart guide [here](https://ai2-climate-emulator.readthedocs.io/en/latest/quickstart.html).

## Model checkpoints

Pretrained model checkpoints are available in the [ACE Hugging Face](https://huggingface.co/collections/allenai/ace-67327d822f0f0d8e0e5e6ca4) collection.

## Available datasets
Two versions of the complete dataset described in [arxiv:2310.02074](https://arxiv.org/abs/2310.02074)
are available on a [requester pays](https://cloud.google.com/storage/docs/requester-pays) Google Cloud Storage bucket:
```
gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-zarrs
gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs
```
The `zarr` format is convenient for ad-hoc analysis. The netCDF version contains our
train/validation split which was used for training and inference.

The datasets used in the [ACE2 paper](https://arxiv.org/abs/2411.11268) are available at:
```
gs://ai2cm-public-requester-pays/2024-11-13-ai2-climate-emulator-v2-amip/data/c96-1deg-shield/
gs://ai2cm-public-requester-pays/2024-11-13-ai2-climate-emulator-v2-amip/data/era5-1deg-1940-2022.zarr/
```

The dataset used in the [ACE2-SOM paper](https://arxiv.org/abs/2412.04418) is available at:
```
gs://ai2cm-public-requester-pays/2024-12-05-ai2-climate-emulator-v2-som/SHiELD-SOM-C96
```
