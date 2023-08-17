FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV FME_DIR=/full-model
ENV DGLBACKEND=pytorch

# Install gcloud
# https://cloud.google.com/sdk/docs/install#deb
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

# install python deps
COPY requirements_except_torch.txt /tmp/requirements.txt
COPY requirements_no_deps.txt /tmp/requirements_no_deps.txt
RUN python3 -m pip install -r /tmp/requirements.txt
RUN python3 -m pip install --no-deps -r /tmp/requirements_no_deps.txt

# copy local code
COPY models/FourCastNet ${FME_DIR}/models/FourCastNet
COPY models/fcn-mip ${FME_DIR}/models/fcn-mip
COPY fme ${FME_DIR}/fme
COPY install_local_packages.sh ${FME_DIR}/install_local_packages.sh

# install packages in full-model repo
RUN cd $FME_DIR && ./install_local_packages.sh
