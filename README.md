[![Docs](https://readthedocs.org/projects/ai2-climate-emulator/badge/?version=latest)](https://ai2-climate-emulator.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/fme.svg)](https://pypi.org/project/fme/)
[![Model checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow)](https://huggingface.co/collections/allenai/ace)

<img src="ACE-logo.png" alt="Logo for the ACE Project" style="width: auto; height: 50px;">

# Ai2 Climate Emulator

> **⚠️ IMPORTANT MIGRATION NOTICE**
>
> This repository had a **breaking history change** on the `main` branch in December 2025 as part of our transition to open development. If you have an existing clone from before this migration, you will need to take action.
>
> **See [MIGRATION.md](MIGRATION.md) for complete instructions.**
>
> - If you have no local work to preserve: delete your local clone and re-clone the repository
> - If you have local branches or commits: follow the detailed migration steps in [MIGRATION.md](MIGRATION.md)

Ai2 Climate Emulator (ACE) is a fast machine learning model that simulates global atmospheric variability in a changing climate over time scales ranging from hours to centuries. This repository contains the `fme` python package which can be used to train, run and evaluate weather and climate AI models such as ACE. It also contains the data processing scripts and model configurations used in recent papers published by the Ai2 Climate Modeling group.

## Installation

```
pip install fme
```

## Documentation

See complete documentation [here](https://ai2-climate-emulator.readthedocs.io/en/latest/) and a quickstart guide [here](https://ai2-climate-emulator.readthedocs.io/en/latest/quickstart.html).

## Model checkpoints

Pretrained model checkpoints are available in the [ACE Hugging Face](https://huggingface.co/collections/allenai/ace-67327d822f0f0d8e0e5e6ca4) collection.

## Papers

The following papers described models trained using code in this repository.

- "ACE: A fast, skillful learned global atmospheric model for climate prediction" ([link](https://arxiv.org/abs/2310.02074))
- "Application of the Ai2 Climate Emulator to E3SMv2's global atmosphere model, with a focus on precipitation fidelity" ([link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JH000136))
- "ACE2: Accurately learning subseasonal to decadal atmospheric variability and forced responses" ([link](https://www.nature.com/articles/s41612-025-01090-0))
- "ACE2-SOM: Coupling an ML Atmospheric Emulator to a Slab Ocean and Learning the Sensitivity of Climate to Changed CO<sub>2</sub>" ([link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JH000575))
- "Applying the ACE2 Emulator to SST Green's Functions for the E3SMv3 Global Atmosphere Model" ([link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000774))
- "SamudrACE: Fast and Accurate Coupled Climate Modeling with 3D Ocean and Atmosphere Emulators" ([link](https://arxiv.org/abs/2509.12490))
- "HiRO-ACE: Fast and skillful AI emulation and downscaling trained on a 3 km global storm-resolving model" ([link](https://arxiv.org/abs/2512.18224))
