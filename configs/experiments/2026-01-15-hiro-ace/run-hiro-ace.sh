# First run coarse ACE2S
python -m fme.ace.inference inference_config.yaml

# Downscale ACE2S using HiRO

# HiRO is more computationally costly then ACE2S
# for faster through put more GPUs may be required
NGPU=1

torchrun --nproc_per_node $NGPU -m fme.downscaling.inference hiro_downscaling_ace2s_global_output.yaml
