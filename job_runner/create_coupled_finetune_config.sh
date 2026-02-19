#!/bin/bash

set -e

EXISTING_RESULTS_DATASET=${1}
TEMPLATE_CONFIG_PATH=${2}
CONFIG_PATH=${3}

beaker dataset fetch ${EXISTING_RESULTS_DATASET} --prefix config.yaml

cp $TEMPLATE_CONFIG_PATH $CONFIG_PATH

# NOTE: requires yq >= 4

# update the coupled stepper config, removing training-specific fields from each component
yq -i '.stepper *= (load("config.yaml").stepper | del(.ocean) | del(.atmosphere))' $CONFIG_PATH
yq -i '.stepper.ocean.stepper *= (load("config.yaml").stepper.ocean.stepper | del(.loss, .optimize_last_step_only, .n_ensemble, .parameter_init, .train_n_forward_step))' $CONFIG_PATH
yq -i '.stepper.atmosphere.stepper *= (load("config.yaml").stepper.atmosphere.stepper | del(.loss, .optimize_last_step_only, .n_ensemble, .parameter_init, .train_n_forward_step))' $CONFIG_PATH

# cleanup
rm ./config.yaml
