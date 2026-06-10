#!/bin/bash

set -e

EXISTING_RESULTS_DATASET=${1}
TEMPLATE_CONFIG_PATH=${2}
CONFIG_PATH=${3}

beaker dataset fetch ${EXISTING_RESULTS_DATASET} --prefix config.yaml

cp $TEMPLATE_CONFIG_PATH $CONFIG_PATH

# NOTE: requires yq >= 4

# Extract loss weights from existing config before removing training fields.
# Prefer stepper_training.loss.weights (newer format), fall back to stepper.loss.weights.
yq '(.stepper_training.loss.weights // .stepper.loss.weights) | select(. != null)' \
    ./config.yaml > ./loss_weights.yaml

# Remove training-specific fields from loaded stepper configs
yq -i 'del(.stepper.loss, .stepper.optimize_last_step_only, .stepper.n_ensemble, .stepper.parameter_init, .stepper.train_n_forward_steps)' ./config.yaml

# update stepper config, preserving template values on conflict
yq -i '.stepper *=n load("config.yaml").stepper' $CONFIG_PATH

# Inject loss weights from existing config if they exist and the template
# doesn't already define them
if [[ -s ./loss_weights.yaml ]]; then
    HAS_TEMPLATE_WEIGHTS=$(yq '.stepper_training.loss.weights | . != null' $CONFIG_PATH)
    if [[ "$HAS_TEMPLATE_WEIGHTS" != "true" ]]; then
        yq -i '.stepper_training.loss.weights = load("loss_weights.yaml")' $CONFIG_PATH
    fi
fi

# cleanup
rm -f ./config.yaml ./loss_weights.yaml
