[![Docs](https://readthedocs.org/projects/ai2-climate-emulator/badge/?version=latest)](https://ai2-climate-emulator.readthedocs.io/en/latest/)

# ACE: Ai2 Climate Emulator
This repo contains code accompanying "ACE: A fast, skillful learned global atmospheric model for climate prediction" ([arxiv:2310.02074](https://arxiv.org/abs/2310.02074)).

## Documentation

See complete documentation [here](https://ai2-climate-emulator.readthedocs.io/en/latest/).

## Quickstart

A quickstart guide may be found [here](https://ai2-climate-emulator.readthedocs.io/en/latest/quickstart.html).

## Available datasets
Two versions of the dataset described in [arxiv:2310.02074](https://arxiv.org/abs/2310.02074)
are available:
```
gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-zarrs
gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs
```
The `zarr` format is convenient for ad-hoc analysis. The netCDF version contains our
train/validation split which was used for training and inference.
