import fme

"""
Manually triggered for CI tests on GPU so that tests do not
default to CPU if driver issues prevent use of CUDA.
"""
device = str(fme.get_device())
print(f"Device: {device}")
assert device.startswith("cuda")
