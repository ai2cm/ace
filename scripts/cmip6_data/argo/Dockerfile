FROM condaforge/mambaforge:latest

ENV DEBIAN_FRONTEND=noninteractive

# The mambaforge image may be missing libxml2; install it so mamba works.
RUN apt-get update && apt-get install -y libxml2 && rm -rf /var/lib/apt/lists/*

# Install xesmf + esmf via conda-forge first, before any apt packages
# that might interfere with conda's shared libraries.
RUN mamba install -y -c conda-forge xesmf esmf && mamba clean -afy

# Install remaining Python dependencies via pip.
COPY argo/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# System packages (gcloud CLI for GCS access).
RUN apt-get update && apt-get install -y \
    apt-transport-https ca-certificates gnupg curl \
    && rm -rf /var/lib/apt/lists/*

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install google-cloud-cli -y && \
    rm -rf /var/lib/apt/lists/*

RUN gcloud config set project vcm-ml

# Copy all pipeline scripts into the container.
# Build context should be scripts/cmip6_data/.
COPY *.py /app/
WORKDIR /app
