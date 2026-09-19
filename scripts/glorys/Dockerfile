FROM continuumio/miniconda3:24.1.2-0

# Only ca-certificates (HTTPS to CloudFerro and GCS) and libgomp1 (ESMF's
# OpenMP runtime) are used here. apt-transport-https, gnupg and curl were also
# installed but nothing in the image needs them: no third-party apt repository
# is added, and all HTTP goes through fsspec/aiohttp. Their dependency tree is
# what broke the build -- bullseye-security no longer serves the
# 2.2.27-2+deb11u3 gnupg revisions its package index still advertises, so all
# of them 404.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN conda create -y -n env python=3.11 \
    && conda install -y -c conda-forge -n env esmpy "xesmf>=0.8"

COPY dataflow-requirements.txt /tmp/requirements.txt
RUN pip install uv \
    && uv pip install --python=/opt/conda/envs/env/bin/python -r /tmp/requirements.txt

COPY --from=apache/beam_python3.11_sdk:2.71.0 /opt/apache/beam /opt/apache/beam
ENTRYPOINT [ "/opt/apache/beam/boot" ]

ENV PATH /opt/conda/envs/env/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/conda/envs/env/lib:${LD_LIBRARY_PATH}
ENV ESMFMKFILE /opt/conda/envs/env/lib/esmf.mk
