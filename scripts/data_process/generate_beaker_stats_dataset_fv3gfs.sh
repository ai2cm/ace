#!/bin/bash

set -e

while [[ "$#" -gt 0 ]]
do case $1 in
    --input-url) INPUT_URL="$2"
    shift;;
    --start-date) START_DATE="$2"
    shift;;
    --end-date) END_DATE="$2"
    shift;;
    --name) DATASET_NAME="$2"
    shift;;
    --desc) DATASET_DESC="$2"
    shift;;
    --script-flags) SCRIPT_FLAGS="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

if [[ -z "${INPUT_URL}" ]]
then
    echo "Option --input-url missing"
    exit 1;
elif [[ -z "${START_DATE}" ]]
then
    echo "Option --start-date missing"
    exit 1;
elif [[ -z "${END_DATE}" ]]
then
    echo "Option --end-date missing"
    exit 1;
elif [[ -z "${DATASET_NAME}" ]]
then
    echo "Option --dataset-name missing"
    exit 1;
fi

OUTPUT_DIR="/tmp/$(uuidgen)"

python get_stats.py \
    $INPUT_URL \
    ${OUTPUT_DIR} \
    --start-date $START_DATE \
    --end-date $END_DATE ${SCRIPT_FLAGS}

beaker dataset create ${OUTPUT_DIR} \
    --name ${DATASET_NAME} --desc "${DATASET_DESC}"

rm -rf ${OUTPUT_DIR}
