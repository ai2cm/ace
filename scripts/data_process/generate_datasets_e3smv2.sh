#!/bin/bash -l

#SBATCH -A m4331
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH --output=joblogs/%x-%j.out

while [[ "$#" -gt 0 ]]
do case $1 in
    -i|--input-dir) INPUT_DIR="$2"
    shift;;
    -c|--config) CONFIG="$2"
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
elif [[ -z "${CONFIG}" ]]
then
    echo "Option -c, --config missing"
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
python -u compute_dataset_e3smv2.py --n-workers=16 --config=${CONFIG} \
    -i ${INPUT_DIR} -o ${ZARR}

# Train on first year (intended for training)
python -u convert_to_monthly_netcdf.py \
    ${ZARR} \
    ${OUTPUT_DIR}/traindata \
    --start-date 1970-01-01 \
    --end-date 1970-12-31 \
    --nc-format NETCDF4

# Validation on next 6 months
python -u convert_to_monthly_netcdf.py \
    ${ZARR} \
    ${OUTPUT_DIR}/validdata \
    --start-date 1971-01-01 \
    --end-date 1971-05-31 \
    --nc-format NETCDF4

# Final 6 months for preditiondata reference
python -u convert_to_monthly_netcdf.py \
    ${ZARR} \
    ${OUTPUT_DIR}/predictiondata \
    --start-date 1971-06-01 \
    --end-date 1971-12-31 \
    --nc-format NETCDF4

# compute all stats on training data
python -u get_stats.py ${CONFIG} 0
