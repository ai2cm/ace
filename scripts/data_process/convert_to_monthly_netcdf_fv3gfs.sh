#!/bin/bash

# This script launches N_IC jobs to convert all GCS zarr data to local monthly netCDFs

while [[ "$#" -gt 0 ]]
do case $1 in
    --input-url) BASE_INPUT_URL="$2"
    shift;;
    --n-ic) N_IC=$2
    shift;;
    --output-dir) BASE_OUTPUT_DIR="$2"
    shift;;
    --start-date) START_DATE="$2"
    shift;;
    --end-date) END_DATE="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

if [[ -z "${BASE_INPUT_URL}" ]]
then
    echo "Option --input-url missing"
    exit 1;
elif [[ -z "${N_IC}" ]]
then
    echo "Option --n-ic missing"
    exit 1;
elif [[ -z "${BASE_OUTPUT_DIR}" ]]
then
    echo "Option --output-dir missing"
    exit 1;
elif [[ -z "${START_DATE}" ]]
then
    echo "Option --start-date missing"
    exit 1;
elif [[ -z "${END_DATE}" ]]
then
    echo "Option --end-date missing"
    exit 1;
fi



for IC in $(seq 1 $(( N_IC ))); do
    IC_STR=$(printf "%04d" ${IC})
    INPUT_URL=${BASE_INPUT_URL}/ic_${IC_STR}.zarr
    OUTPUT_DIR=${BASE_OUTPUT_DIR}/ic_${IC_STR}
    python convert_to_monthly_netcdf.py \
        $INPUT_URL \
        $OUTPUT_DIR \
        --start-date $START_DATE \
        --end-date $END_DATE &
done

wait  # Wait for all background processes to complete
