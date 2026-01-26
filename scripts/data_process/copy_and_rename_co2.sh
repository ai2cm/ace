#!/bin/bash
# rename_co2_variables.sh
# Loop through CM4 Zarr files and rename carbon_dioxide to global_mean_co2

# Set path to your Python rename script
PYTHON_SCRIPT="/path/to/copy_and_rename_variable_inplace.py"

# Array of full paths to the Zarr stores
ZARR_FILES=(
    "/climate-default/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr"
    "/climate-default/2025-06-18-CM4-1pctCO2-atmosphere-land-1deg-8layer-140yr.zarr"
    "/climate-default/2025-12-02-CM4-like-AM4-random-CO2/random-CO2-1xCO2-ic_0001.zarr"
    "/climate-default/2025-12-02-CM4-like-AM4-random-CO2/random-CO2-2xCO2-ic_0001.zarr"
    "/climate-default/2025-12-02-CM4-like-AM4-random-CO2/random-CO2-4xCO2-ic_0001.zarr"
    "/climate-default/2025-12-02-CM4-like-AM4-random-CO2/random-CO2-1xCO2-ic_0002.zarr"
    "/climate-default/2025-12-02-CM4-like-AM4-random-CO2/random-CO2-2xCO2-ic_0002.zarr"
    "/climate-default/2025-12-02-CM4-like-AM4-random-CO2/random-CO2-4xCO2-ic_0002.zarr"
)

# Loop through each Zarr file and run the Python script
for zarr_file in "${ZARR_FILES[@]}"; do
    echo "Processing $zarr_file ..."
    python "$PYTHON_SCRIPT" \
        --store-path "$zarr_file" \
        --variable-name "carbon_dioxide" \
        --new-name "global_mean_co2"
done

echo "All files processed."
