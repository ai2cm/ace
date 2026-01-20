# Example script to show the steps needed to downscale ACE
# using HiRO

# Provided are two inference configurations used in the
# manuscript:
# 1.) Global inference with ACE for 2023 then downscaled
# for most of the global (-65S to 65N) using HiRO
#
# 2.) Global inference with ACE for 2014 - 2023 then
# the Pacific NorthWest (PNW) only downscaled with HiRO


# First run coarse ACE2S
python -m fme.ace.inference ace2s_inference_config_global.yaml

# Second downscale ACE2S using HiRO

# HiRO is more computationally costly then ACE2S
# for faster through put more GPUs may be required
NGPU=1

torchrun --nproc_per_node $NGPU -m fme.downscaling.inference hiro_downscaling_ace2s_global_output.yaml
