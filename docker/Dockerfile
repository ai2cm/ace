FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime AS base

ENV FME_DIR=/full-model
ENV DGLBACKEND=pytorch
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

RUN apt-get update -y && apt-get install -y \
  build-essential \
  curl \
  git \
  gnupg \
  wget

# Install gcloud- used for monthly netcdf data processing script
# https://cloud.google.com/sdk/docs/install#deb
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

ENV FORCE_CUDA_EXTENSION=1
# install python deps
RUN pip install uv
COPY pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system -r /tmp/pyproject.toml --extra graphcast
COPY requirements-healpix.txt /tmp/requirements-healpix.txt
RUN python3 -m pip install --no-build-isolation -r /tmp/requirements-healpix.txt

# set default matplotlib cmap
RUN mkdir -p $HOME/.config/matplotlib && \
  echo "image.cmap: turbo" >> $HOME/.config/matplotlib/matplotlibrc

FROM base AS production

# copy local code and install
COPY . ${FME_DIR}
RUN cd $FME_DIR && pip install --no-deps -e .

# copy after install so editing scripts does not trigger reinstall
COPY scripts ${FME_DIR}/scripts

FROM base AS deps-only

# install github cli and yq
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update && apt-get install -y gh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq &&\
chmod +x /usr/bin/yq

COPY docker/install_apex_groupnorm.sh /tmp/install_apex_groupnorm.sh
RUN chmod +x /tmp/install_apex_groupnorm.sh && /tmp/install_apex_groupnorm.sh && rm /tmp/install_apex_groupnorm.sh

ENTRYPOINT ["/bin/bash"]

FROM deps-only AS nsight

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y -q --no-install-recommends --no-install-suggests \
  cmake \
  libglib2.0 \
  && rm -rf /var/lib/apt/lists/*

# install Nsight Systems
ARG NSYS_DEB=NsightSystems-linux-cli-public-2025.5.1.121-3638078.deb

RUN wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_5/${NSYS_DEB} \
    && dpkg -i ${NSYS_DEB} \
    && rm ${NSYS_DEB}

ENTRYPOINT ["/bin/bash"]
CMD ["-l"]
