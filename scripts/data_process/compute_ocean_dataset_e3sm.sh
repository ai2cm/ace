#!/bin/bash

# Three steps to preprocess E3SM ocean and sea ice data:
# 1. Vertical coarsening
# 2. Remap to 1 degree grid with ncremap
# 3. Run standard ocean preprocessing

make create_ocean_data_proc_env
conda activate ocean_preprocess

# Step 1: Vertical coarsening
year=401
vertical_coarsening_config=configs/E3SMv3-piControl-ocean-vertical-coarsen.yaml
python -u e3sm_ocean_vertical_coarsen.py --year="$year" --config=$vertical_coarsening_config

# Step 2: Remap to 1 degree grid with ncremap
source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.sh

coarsend_output_path=$(sed -n 's/^output_path: //p' $vertical_coarsening_config)
coarsened_output_prefix=$(sed -n 's/^output_prefix: //p' $vertical_coarsening_config)

INPUT_DIR=$coarsend_output_path
OUTPUT_DIR=$PSCRATCH/v3.LR.piControl_bonus/vertical_coarsened/1deg_test
GRID_FILE=/global/cfs/cdirs/m4492/e3sm-couple-run/maps/map_IcoswISC30E3r5_to_gaussian_180x360.nc
FILES=($INPUT_DIR/$coarsened_output_prefix*.nc)
TMP_DIR=$PSCRATCH/tmp
export GRID_FILE
export OUTPUT_DIR
SLURM_NTASKS=16
echo "Running with $SLURM_NTASKS jobs"
parallel --jobs $SLURM_NTASKS --tmpdir $TMP_DIR \
    'ncremap -P mpas -i {} -o $OUTPUT_DIR/$(basename {}) --map=$GRID_FILE' ::: "${FILES[@]}"

# Step 3: Run standard ocean preprocessing
conda activate ocean_preprocess
standard_ocean_preprocessing_config=configs/e3smv3-ocean-1deg.yaml
output_store=$PSCRATCH/fme_dataset/2025-11-13-E3SMv3-piControl-ocean-ice-1yr-subsample.zarr
python -u compute_ocean_dataset_e3sm.py \
        --config="${standard_ocean_preprocessing_config}" \
        --output-store="${output_store}" \
        --subsample

python -u get_stats.py "${standard_ocean_preprocessing_config}" 0
