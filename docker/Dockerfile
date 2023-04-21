FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV FME_DIR=/full-model

# update repo info
RUN apt update -y

# install python deps
RUN python3 -m pip install wandb \
    mpi4py \
    h5py \
    ruamel.yaml \
    tqdm \
    timm \
    einops \
    moviepy \
    imageio \
    netCDF4 \
    s3fs \
    git+https://github.com/tensorly/tensorly \
    git+https://github.com/tensorly/torch

# install FourCastNet package
COPY models/FourCastNet ${FME_DIR}/models/FourCastNet
RUN python3 -m pip install -e ${FME_DIR}/models/FourCastNet

# install fcn_mip package and some dependencies
COPY models/fcn-mip ${FME_DIR}/models/fcn-mip
RUN python3 -m pip install --no-deps -e ${FME_DIR}/models/fcn-mip
RUN python3 -m pip install --no-deps -e ${FME_DIR}/models/fcn-mip/external/torch_sht
RUN python3 -m pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html && \
    python3 -m pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
ENV DGLBACKEND=pytorch
RUN python3 -m pip install -e ${FME_DIR}/models/fcn-mip/external/modulus-core

# install fme package
COPY fme ${FME_DIR}/fme
RUN python3 -m pip install -e ${FME_DIR}/fme
