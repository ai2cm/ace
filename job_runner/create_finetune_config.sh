#!/bin/bash

set -e

EXISTING_RESULTS_DATASET=${1}
TEMPLATE_CONFIG_PATH=${2}
CONFIG_PATH=${3}

beaker dataset fetch ${EXISTING_RESULTS_DATASET} --prefix config.yaml

cp $TEMPLATE_CONFIG_PATH $CONFIG_PATH

# NOTE: requires yq >= 4

# Remove training-specific fields from loaded stepper configs
yq -i 'del(.stepper.loss, .stepper.optimize_last_step_only, .stepper.n_ensemble, .stepper.parameter_init, .stepper.train_n_forward_step)' ./config.yaml

# update stepper config, preserving template values on conflict
yq -i '.stepper *=n load("config.yaml").stepper' $CONFIG_PATH

# cleanup
rm ./config.yaml
