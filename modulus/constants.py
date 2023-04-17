"""
constant values used by Modulus
"""

import torch
import numpy as np

# string used to determine derivatives
diff_str: str = "__"

# function to apply diff string
def diff(y: str, x: str, degree: int = 1) -> str:
    return diff_str.join([y] + degree * [x])


# for changing to float16 or float64
tf_dt = torch.float32
np_dt = np.float32

# tensorboard naming
TF_SUMMARY = False

# Pytorch Version for which JIT will be default on
# Torch version of NGC container 22.08
JIT_PYTORCH_VERSION = "1.13.0a0+d321be6"

# No scaling is needed if using NO_OP_SCALE
NO_OP_SCALE = (0.0, 1.0)

# If using NO_OP_NORM, it is effectively doing no normalization
NO_OP_NORM = (-1.0, 1.0)
