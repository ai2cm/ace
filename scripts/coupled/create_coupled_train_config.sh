#!/bin/bash

set -e

EXISTING_RESULTS_ATMOS_DATASET=${1}
EXISTING_RESULTS_OCEAN_DATASET=${2}
TEMPLATE_CONFIG_PATH=${3}
CONFIG_PATH=${4}

beaker dataset fetch ${EXISTING_RESULTS_ATMOS_DATASET} --prefix config.yaml
mv ./config.yaml ./atmos-config.yaml
sed -i 's/statsdata/atmos_stats/g' ./atmos-config.yaml

beaker dataset fetch ${EXISTING_RESULTS_OCEAN_DATASET} --prefix config.yaml
mv ./config.yaml ./ocean-config.yaml
sed -i 's/statsdata/ocean_stats/g' ./ocean-config.yaml

# try to get sea_ice_fraction_name from step config
SIC_NAME=$(yq '.stepper.step.config.corrector.config.sea_ice_fraction_correction.sea_ice_fraction_name' ./ocean-config.yaml)

if [[ "$SIC_NAME" == "null" ]]; then
    echo "Failed to extract sea_ice_fraction_name from the ocean config"
    exit 1
fi

cp $TEMPLATE_CONFIG_PATH $CONFIG_PATH

# NOTE: requires yq >= 4

# update ocean stepper config
yq -i '.stepper.ocean.stepper *= load("ocean-config.yaml").stepper' $CONFIG_PATH
SIC_NAME=$SIC_NAME yq -i '.stepper.ocean_fraction_prediction.sea_ice_fraction_name = env(SIC_NAME)' $CONFIG_PATH

# update atmos stepper config
yq -i '.stepper.atmosphere.stepper *= load("atmos-config.yaml").stepper' $CONFIG_PATH

# cleanup
rm ./atmos-config.yaml ./ocean-config.yaml
