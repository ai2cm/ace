#!/bin/bash -l

#SBATCH -A m4331
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --image=registry.services.nersc.gov/ai2cm/fv3net:latest

while [[ "$#" -gt 0 ]]
do case $1 in
    -i|--input-dir) INPUT_DIR="$2"
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
    echo "Option -i,--input-dir missing"
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

mkdir -p $OUTPUT_DIR

set -xe

# create the zarr from E3SMv2 nc files
srun -u shifter \
    bash -c "
        python compute_dataset_e3smv2.py --n-split=100 --n-workers=32 \
            -i ${INPUT_DIR} -o ${ZARR}
    "

# save first 19 years of data to netCDFs (intended for training)
srun -u shifter \
    bash -c "
        python convert_to_monthly_netcdf.py \
            ${ZARR} \
            ${OUTPUT_DIR}/traindata \
            --start-date 0002-01-01 \
            --end-date 0020-12-31
    "

# save final year of data to netCDFs (intended for validation)
srun -u shifter \
    bash -c "
        python convert_to_monthly_netcdf.py \
            ${ZARR} \
            ${OUTPUT_DIR}/validdata \
            --start-date 0021-01-01 \
            --end-date 0021-12-31
    "

# compute residual scaling stats
srun -u shifter \
    bash -c "
        python get_stats.py \
            ${ZARR} \
            ${OUTPUT_DIR}/statsdata/residual \
            --start-date 0002-01-01 \
            --end-date 0020-12-31 \
            --data-type E3SMV2
    "

# compute full field scaling stats
srun -u shifter \
    bash -c "
        python get_stats.py \
            ${ZARR} \
            ${OUTPUT_DIR}/statsdata/full_field \
            --start-date 0002-01-01 \
            --end-date 0020-12-31 \
            --data-type E3SMV2
            --full-field
    "
