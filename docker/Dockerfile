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
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

FROM base AS production

# copy local code and install
COPY . ${FME_DIR}
RUN cd $FME_DIR && pip install --no-deps -e .

# copy after install so editing scripts does not trigger reinstall
COPY scripts ${FME_DIR}/scripts