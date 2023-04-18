FROM nvcr.io/nvidia/pytorch:21.11-py3

ENV FME_DIR=/full-model

# update repo info
RUN apt update -y

# install python deps
RUN pip install wandb && \
    pip install mpi4py && \
    pip install h5py && \
    pip install ruamel.yaml && \
    pip install --upgrade tqdm && \
    pip install timm && \
    pip install einops && \
    pip install moviepy imageio && \
    pip install netCDF4

# install FourCastNet package
COPY models/FourCastNet ${FME_DIR}/models/FourCastNet
RUN python3 -m pip install -e ${FME_DIR}/models/FourCastNet

# install fme package
COPY fme ${FME_DIR}/fme
RUN python3 -m pip install -e ${FME_DIR}/fme
