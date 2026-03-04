FROM continuumio/miniconda3:24.1.2-0

RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates gnupg curl

RUN conda create -y -n env python=3.11 \
    && conda install -y -c conda-forge -n env esmf xesmf

COPY dataflow-requirements.txt /tmp/requirements.txt
RUN pip install uv \
    && uv pip install --python=/opt/conda/envs/env/bin/python -r /tmp/requirements.txt

COPY --from=apache/beam_python3.11_sdk:2.71.0 /opt/apache/beam /opt/apache/beam
ENTRYPOINT [ "/opt/apache/beam/boot" ]

ENV PATH /opt/conda/envs/env/bin:${PATH}
# Normally this variable would be set by activating the conda environment, but
# since we are not doing that here, we need to set it manually. It is required
# for xesmf to be able to find esmf. See
# https://github.com/conda-forge/esmf-feedstock/issues/91#issuecomment-1387279692
ENV ESMFMKFILE /opt/conda/envs/env/lib/esmf.mk
