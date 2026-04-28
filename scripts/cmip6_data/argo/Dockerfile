FROM condaforge/mambaforge:latest

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

# xesmf + esmf must come from conda-forge (ESMF is a compiled
# Fortran/C library with no pip-installable binary).
RUN mamba install -y -c conda-forge xesmf esmf && mamba clean -afy

# Install remaining Python dependencies via pip.
COPY argo/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy all pipeline scripts into the container.
# Build context should be scripts/cmip6_data/.
COPY *.py /app/
WORKDIR /app
