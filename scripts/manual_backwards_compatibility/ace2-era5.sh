#!/bin/bash

set -e

# This script can be used to ensure that your currently installed version of the fme
# package can do inference with the published ACE2-ERA5 model.

# download necessary data
mkdir -p test_inference_ace2_era5
cd test_inference_ace2_era5
mkdir -p initial_conditions
mkdir -p forcing_data
wget https://huggingface.co/allenai/ACE2-ERA5/resolve/main/ace2_era5_ckpt.tar?download=true -O ace2_era5_ckpt.tar
wget https://huggingface.co/allenai/ACE2-ERA5/resolve/main/inference_config.yaml?download=true -O inference_config.yaml
wget https://huggingface.co/allenai/ACE2-ERA5/resolve/main/initial_conditions/ic_2020.nc?download=true -O initial_conditions/ic_2020.nc
wget https://huggingface.co/allenai/ACE2-ERA5/resolve/main/forcing_data/forcing_2020.nc?download=true -O forcing_data/forcing_2020.nc

# update config to use relative paths and do a short run
yq e '.n_forward_steps = 50' -i inference_config.yaml
yq e '.forward_steps_in_memory = 5' -i inference_config.yaml
yq e '.checkpoint_path = "ace2_era5_ckpt.tar"' -i inference_config.yaml
yq e '.initial_condition.path = "initial_conditions/ic_2020.nc"' -i inference_config.yaml
yq e '.forcing_loader.dataset.data_path = "forcing_data/"' -i inference_config.yaml

# run on CPU or CUDA if the latter is available
yq e '.experiment_dir = "output_cpu"' -i inference_config.yaml
python -m fme.ace.inference inference_config.yaml

# run on MPS. NOTE: this requires torch==2.5 otherwise there are complaints about some of the
# features used by the SFNO architecture.
yq e '.experiment_dir = "output_mps"' -i inference_config.yaml
export FME_USE_MPS=1
python -m fme.ace.inference inference_config.yaml
