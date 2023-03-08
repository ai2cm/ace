FROM nvcr.io/nvidia/pytorch:21.11-py3

# update repo info
RUN apt update -y

ENV TZ="America/Los_Angeles"

# Install base tools.
RUN apt-get update && apt-get install -y \
#    build-essential \
#    curl \
#    git \
#    jq \
#    language-pack-en \
#    make \
#    man-db \
#    manpages \
#    manpages-dev \
#    manpages-posix \
#    manpages-posix-dev \
    sudo #\
#    unzip \
#    vim \
#    wget \
#    fish \
#    parallel \
#    iputils-ping \
#    htop \
#    emacs \
#    tmux \
#    libglvnd0 \
#    libgl1 \
#    libglx0 \
#    libegl1 \
#    libgles2 \
#    libxcb1-dev \
#    vulkan-utils

# install mpi4py
RUN pip install mpi4py

# h5py
RUN pip install h5py

# other python stuff
RUN pip install wandb && \
    pip install ruamel.yaml && \
    pip install --upgrade tqdm && \
    pip install timm && \
    pip install einops && \
    pip install moviepy imageio && \
    pip install netCDF4

# benchy
RUN pip install git+https://github.com/romerojosh/benchy.git

# set wandb to offline
#ENV WANDB_MODE offline

# copy source code
RUN mkdir -p /opt/ERA5_wind
COPY config /opt/ERA5_wind/config
COPY copernicus /opt/ERA5_wind/copernicus
COPY docker /opt/ERA5_wind/docker
COPY networks /opt/ERA5_wind/networks
COPY utils /opt/ERA5_wind/utils
COPY inference /opt/ERA5_wind/inference
COPY *.py /opt/ERA5_wind/
COPY *.sh /opt/ERA5_wind/

# create dummy git image
RUN cd /opt/ERA5_wind && git init

#### https://github.com/allenai/docker-images/blob/main/cuda/Dockerfile ####

# Ensure users can modify their container environment.
RUN echo '%users ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Make the base image friendlier for interactive workloads. This makes things like the man command
# work.
RUN yes | unminimize

# Shell customization including prompt and colors.
###### Broken
# COPY profile.d/ /etc/profile.d/

# TODO(gideond) this still hangs when you try to install the full deps. For
# now, just remove them I think.

ENTRYPOINT ["bash", "-l"]
