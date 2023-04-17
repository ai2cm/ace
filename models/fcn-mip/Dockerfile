ARG PYT_VER=22.08
FROM nvcr.io/nvidian/pytorch:$PYT_VER-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    fuse \
    gettext \
    awscli \
    parallel \
    libgeos-dev \
    cdo \
    nco \
    netcdf-bin \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Need to install in the system python3 package
RUN /usr/bin/python3 -m pip install awscli-plugin-endpoint

COPY docker/install_mamba.sh /tmp/install_mamba.sh
RUN bash /tmp/install_mamba.sh

# pangeo stuff
RUN mamba install -y xarray dask metpy distributed h5py kerchunk cartopy s3fs

RUN python3 -m pip install \
    dask-mpi  \
    pangeo_forge_recipes \
    pre-commit \
    wandb \
    ruamel-yaml \
    timm \
    einops \
    git+https://github.com/tensorly/tensorly \
    git+https://github.com/tensorly/torch \
    'bokeh<3'

# Install git-lfs
COPY bin/add_package_cloud_apt.sh /tmp/
RUN /tmp/add_package_cloud_apt.sh && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*

# Install xpartition
COPY external/torch_sht/ /opt/torch_sht
RUN python3 -m pip install -e /opt/torch_sht

# Install modulus-core
COPY external/modulus-core/ /opt/modulus-core
RUN pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html
RUN pip install -r /opt/modulus-core/dockerfiles/requirements_ci.txt
RUN python3 -m pip install -e /opt/modulus-core

# Install dgl nightly build
# required for bfloat16 support w/ graphcast
RUN pip install --pre dgl -f https://data.dgl.ai/wheels/cu117/repo.html
RUN pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html

ARG GIT_COMMIT_SHA
ENV GIT_COMMIT_SHA=$GIT_COMMIT_SHA

ENV GIT_REPO=https://gitlab-master.nvidia.com/earth-2/fcn-mip

COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]

ENV PASSWORD=mypass
ENV VSCODE_PORT=8899
# will pip install and update submodules
ENV FCN_MIP_SRC=""