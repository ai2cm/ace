#!/bin/bash

set -e

# This script can be used to ensure that your currently installed version of the fme
# package can do inference with the published ACE-climSST model.

# download necessary data
mkdir -p test_inference_ace_climSST
cd test_inference_ace_climSST
wget https://huggingface.co/allenai/ACE-climSST/resolve/main/ace_climSST_ckpt.tar?download=true -O ace_climSST_ckpt.tar
wget https://huggingface.co/allenai/ACE-climSST/resolve/main/inference-config.yaml?download=true -O inference-config.yaml
wget https://huggingface.co/allenai/ACE-climSST/resolve/main/climSST.tar.gz?download=true -O climSST.tar.gz

tar -xvzf climSST.tar.gz

# update config to use relative paths and do a short run
yq e '.n_forward_steps = 50' -i inference-config.yaml
yq e '.forward_steps_in_memory = 5' -i inference-config.yaml
yq e '.checkpoint_path = "ace_climSST_ckpt.tar"' -i inference-config.yaml
yq e '.initial_condition.path = "initial_conditions/ic_2020.nc"' -i inference-config.yaml
yq e '.forcing_loader.dataset.data_path = "forcing_data/"' -i inference-config.yaml

# run on CPU or CUDA if the latter is available
yq e '.experiment_dir = "output_cpu"' -i inference-config.yaml
python -m fme.ace.inference inference-config.yaml
