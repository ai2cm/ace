FROM python:3.12-slim

# install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# install git and git-lfs (needed to install torch_harmonics from GitHub)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    google-cloud-sdk

# install gcloud sdk
RUN gcloud config set project vcm-ml

COPY requirements-atmosphere.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
