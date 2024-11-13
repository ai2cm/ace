#!/bin/bash

mkdir -p ./raw_checkpoints
mkdir -p ./final_checkpoints

# original ACE checkpoint
beaker dataset fetch oliverwm/ace-checkpoint-2023 --output ./raw_checkpoints/ace
python process_ckpt.py \
    --strip-optimization \
    --cast-coords-to-float32 \
    ./raw_checkpoints/ace/ace_ckpt.tar \
    ./final_checkpoints/ace_climSST_ckpt.tar

# E3SM checkpoint
wget https://portal.nersc.gov/archive/home/projects/e3sm/www/e3smv2-fme-dataset/ema_ckpt.tar -O ./raw_checkpoints/ace_climSST_EAMv2_ckpt_full.tar
python process_ckpt.py \
    --strip-optimization \
    --cast-coords-to-float32 \
    ./raw_checkpoints/ace_climSST_EAMv2_kpt_full.tar \
    ./final_checkpoints/ace_climSST_EAMv2_ckpt.tar

# ACE2-ERA5 checkpoint
beaker dataset fetch 01J4MT10JPQ8MFA41F2AXGFYJ9 --prefix training_checkpoints/best_inference_ckpt.tar --output ./raw_checkpoints/ace2_era5
python process_ckpt.py \
    --strip-optimization \
    --cast-coords-to-float32 \
    ./raw_checkpoints/ace2_era5/training_checkpoints/best_inference_ckpt.tar \
    ./final_checkpoints/ace2_era5_ckpt.tar