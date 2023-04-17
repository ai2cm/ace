# FCN-MIP

This repository intercompares several FCN models.

See the Makefile for the commands used to generate certain model outputs.

Reports are hosted here: https://earth-2.gitlab-master-pages.nvidia.com/fcn-mip/


## Requirements

Install the following:

- git lfs for downloading some larger files
- torch_harmonics: https://gitlab.com/nvidia/modulus/torch-harmonics. see git submodule for the specific commit to use.
- modulus-core: https://gitlab.com/nvidia/modulus/modulus. see git submoudle for the specific commit to use.
- See other dependencies in [Dockerfile](./Dockerfile)

## Quick start

### Selene

Start an interactive node on selene:

    ./interactive_selene.sh

Copy in the correct config file

    cp config/selene.py config.py

**important** To avoid stalls due to improper environment variable
*initialization of the distributed manager, you should manually set

    export WORLD_SIZE=1

### NGC

See these
[instuctions](https://confluence.nvidia.com/display/DevtechCompute/NGC+Workflow+for+FCN-MIP)
to get an interactive session in NGC.

Copy in the correct config file

    cp config/ngc/<ace>.py config.py

### Other systems

You will need to setup your own configuration files to point to similar ERA5
data and copy it to `config.py`.

Download the model packages to a directory and point to that directory with
`config.MODEL_REGISTRY`.


### Common

Install git lfs.

Now run the afno inference using make. Under the hood, the makefile use
inference.py. Checkout the makefile and modify as needed.
```
$ make test
```

##  Examples

See [examples](./examples)

## Command line tools

### End-to-end model evaluation

The script `bin/evaluate_model.sh <model>` is intended run an end to end
evaluation pipeline for a given model. (note currently the orchestration of the
ACC/RMSE computation and tcwv are handled in the Makefile, but that should
change soon).

Once this has been done for multiple models, the report can be compiled by running
`python3 plots/report.py`. This will create a static html report at `index.html`
that can be shared.

## Frequently asked questions

### How do I add model?

Download the model packages to a directory and point to that directory with
`config.MODEL_REGISTRY`.

See [this file](./fcn_mip/registry.py) for more details.

## Contributing

- Machine specific configurations (e.g. paths to datasets) should be in
  `config.<machine>.py`
- (optional) install pre-commit hooks and git lfs with `make setup-hooks`.
  After this command is run, you will need to fix any lint errors before
  commiting. This needs to be done once per local clone of this repository.
- Merge requests (contributor instructions)
  - Try to test your code (your reviewer may request you to write some).
  - finish your work and run `make lint test`. Fix any errors that come up.
  - target MRs to main
  - either fork or push a feature branch to this repo directly
  - open the MR, and then slack Yair or Noah to review it.
- Merge requests (reviewer instructions)
  - The reviewer is responsible for merging
  - Clone their code and run `make test lint`
  - Avoid "squash merge"
  - The CI is currently broken. So use the "merge immediately" button.

### Style Guide

Code is read more than it is written. So in this repository we prefer readable
code at the cost of verbosity. The linting helps with this, but we also need
some additional ground rules for things linters cannot flag:

- Naming:
  - Avoid abbreviations even if it seems verbose
  - Avoid synonyms of common concepts (use only one of "sample" and "draw")
