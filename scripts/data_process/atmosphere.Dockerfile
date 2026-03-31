FROM python:3.12-slim

# install google-cloud-cli: https://cloud.google.com/sdk/docs/install#deb
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

# set gcloud project
RUN gcloud config set project vcm-ml

# install git and git-lfs (needed to install torch_harmonics from GitHub)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs

COPY requirements-atmosphere.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
