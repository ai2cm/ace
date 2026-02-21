#!/bin/bash

set -e

EXISTING_RESULTS_DATASET=${1}
TEMPLATE_CONFIG_PATH=${2}
CONFIG_PATH=${3}

beaker dataset fetch ${EXISTING_RESULTS_DATASET} --prefix config.yaml

cp $TEMPLATE_CONFIG_PATH $CONFIG_PATH

# NOTE: requires yq >= 4

# merge checkpoint stepper as base, then overlay with template stepper so that
# any values explicitly set in the template take precedence
yq -i '.stepper = load("config.yaml").stepper * .stepper' $CONFIG_PATH

# cleanup
rm ./config.yaml
