#!/bin/bash -l

#SBATCH -A m4331
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 08:00:00
#SBATCH --output=joblogs/%x-%j.out

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
stripe_small $OUTPUT_DIR

# NOTE: assumes you've already created the fv3net conda env. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310
source activate fv3net

set -xe

# create the zarr from E3SMv2 .nc files
python -u compute_dataset_e3smv2.py --sht-roundtrip \
    --n-split=400 --n-workers=16 \
    -i ${INPUT_DIR} -t ${TIME_INVARIANT_DIR} -o ${ZARR}

# drop first 11 years of data because of stratospheric water drift, and save
# next 42 years of data to netCDFs (intended for training)
python -u convert_to_monthly_netcdf.py \
    ${ZARR} \
    ${OUTPUT_DIR}/traindata \
    --start-date 0012-01-01 \
    --end-date 0053-12-31 \
    --nc-format NETCDF4

# save next 10 years of data to netCDFs (intended for validation)
python -u convert_to_monthly_netcdf.py \
    ${ZARR} \
    ${OUTPUT_DIR}/validdata \
    --start-date 0054-01-01 \
    --end-date 0063-12-31 \
    --nc-format NETCDF4

# save final 10 years of data to netCDFs (intended for prediction_data reference)
python -u convert_to_monthly_netcdf.py \
    ${ZARR} \
    ${OUTPUT_DIR}/predictiondata \
    --start-date 0064-01-01 \
    --end-date 0073-12-31 \
    --nc-format NETCDF4

# compute residual scaling stats on training data
python -u get_stats.py \
    ${ZARR} \
    ${OUTPUT_DIR}/statsdata/residual \
    --start-date 0012-01-01 \
    --end-date 0053-12-31 \
    --data-type E3SMV2

# compute full field scaling stats on training data
python -u get_stats.py \
    ${ZARR} \
    ${OUTPUT_DIR}/statsdata/full_field \
    --start-date 0012-01-01 \
    --end-date 0053-12-31 \
    --data-type E3SMV2 \
    --full-field
