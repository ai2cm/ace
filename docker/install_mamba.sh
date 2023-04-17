#!/bin/bash
# a hacky script to install mamba in a base env with many packages
# conda install is very slow in NVIDIA images due to the number of packages
# installed in the root env

# the python version must match
conda create -y -n mamba -c conda-forge mamba python=3.8

# copy all libraries except libpython
rm -f /opt/conda/envs/mamba/lib/libpython.so*
cp -rf /opt/conda/envs/mamba/lib/* /opt/conda/lib/

# add the mamba script
sed 's|conda/envs/mamba|conda/|' < /opt/conda/envs/mamba/bin/mamba  > /opt/conda/bin/mamba
chmod +x /opt/conda/bin/mamba

