#!/bin/bash -l

#SBATCH -A m4331
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH --output=%x-%j.out

while [[ "$#" -gt 0 ]]
do case $1 in
    -i|--input-dir) INPUT_DIR="$2"
    shift;;
    -t|--time-invariant-input-dir) TIME_INVARIANT_DIR="$2"
    shift;;
    -z|--zarr) ZARR="$2"
    shift;;
    -o|--output-dir) OUTPUT_DIR="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

if [[ -z "${INPUT_DIR}" ]]
then
    echo "Option -i, --input-dir missing"
    exit 1;
elif [[ -z "${TIME_INVARIANT_DIR}" ]]
then
    echo "Option -t, --time-invariant-input-dir missing"
    exit 1;
elif [[ -z "${ZARR}" ]]
then
    echo "Option -z, --zarr missing"
    exit 1;
elif [[ -z "${OUTPUT_DIR}" ]]
then
    echo "Option -o, --output-dir missing"
    exit 1;
fi

# output dir should be somewhere on $SCRATCH, even if the intention is to send
# the data to CFS or some external data store
mkdir -p $OUTPUT_DIR

# stripe_small is recommended for files of size 1-10 GB on Perlmutter's Lustre
# scratch filesystem and stripes across 8 OSTs
# see https://docs.nersc.gov/performance/io/lustre/#nersc-file-striping-recommendations
# TODO: align HDF5 chunks for Perlmutter's default 1MB stripe size
stripe_small $OUTPUT_DIR

# NOTE: assumes you've already created the fv3net conda env
source activate fv3net

set -xe

# create the zarr from E3SMv2 nc files
python -u compute_dataset_e3smv2.py --sht-roundtrip \
    --n-split=200 --n-workers=16 \
    -i ${INPUT_DIR} -t ${TIME_INVARIANT_DIR} -o ${ZARR}

# save first 30 years of data to netCDFs (intended for training)
python -u convert_to_monthly_netcdf.py \
    ${ZARR} \
    ${OUTPUT_DIR}/traindata \
    --start-date 0002-01-01 \
    --end-date 0031-12-31 \
    --nc-format NETCDF4

# save years 0022--0031 to netCDFs (intended for prediction_data baseline)
python -u convert_to_monthly_netcdf.py \
    ${ZARR} \
    ${OUTPUT_DIR}/predictiondata \
    --start-date 0022-01-01 \
    --end-date 0031-12-31 \
    --nc-format NETCDF4

# save final 10 years of data to netCDFs (intended for validation)
python -u convert_to_monthly_netcdf.py \
    ${ZARR} \
    ${OUTPUT_DIR}/validdata \
    --start-date 0032-01-01 \
    --end-date 0041-12-31 \
    --nc-format NETCDF4

# compute residual scaling stats on training data
python -u get_stats.py \
    ${ZARR} \
    ${OUTPUT_DIR}/statsdata/residual \
    --start-date 0002-01-01 \
    --end-date 0031-12-31 \
    --data-type E3SMV2

# compute full field scaling stats on training data
python -u get_stats.py \
    ${ZARR} \
    ${OUTPUT_DIR}/statsdata/full_field \
    --start-date 0002-01-01 \
    --end-date 0031-12-31 \
    --data-type E3SMV2 \
    --full-field
