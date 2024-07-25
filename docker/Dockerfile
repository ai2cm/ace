FROM nvcr.io/nvidia/pytorch:23.08-py3

ENV FME_DIR=/full-model
ENV DGLBACKEND=pytorch

# Install gcloud- used for monthly netcdf data processing script
# https://cloud.google.com/sdk/docs/install#deb
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

# install python deps
COPY fme/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# copy local code and install
COPY fme ${FME_DIR}/fme
RUN cd $FME_DIR && pip install --no-deps -e fme

# copy after install so editing scripts does not trigger reinstall
COPY scripts ${FME_DIR}/scripts
