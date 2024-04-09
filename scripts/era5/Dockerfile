FROM continuumio/miniconda3:24.1.2-0

RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates gnupg curl

RUN conda create -y -n env python=3.9

COPY dataflow-requirements.txt /tmp/requirements.txt
RUN pip install uv \
    && uv pip install --python=/opt/conda/envs/env/bin/python -r /tmp/requirements.txt

COPY --from=apache/beam_python3.9_sdk:2.54.0 /opt/apache/beam /opt/apache/beam
ENTRYPOINT [ "/opt/apache/beam/boot" ]

ENV PATH /opt/conda/envs/env/bin:${PATH}
