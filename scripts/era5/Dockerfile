FROM continuumio/miniconda3:24.1.2-0

RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates gnupg curl

RUN conda create -y -n env python=3.9 \
    && conda install -y -c conda-forge -n env metview-batch

RUN conda install -y -n env -c conda-forge metview-python && conda clean -tip

COPY dataflow-requirements.txt /tmp/requirements.txt
RUN pip install uv \
    && uv pip install --python=/opt/conda/envs/env/bin/python -r /tmp/requirements.txt

COPY --from=apache/beam_python3.9_sdk:2.54.0 /opt/apache/beam /opt/apache/beam
ENTRYPOINT [ "/opt/apache/beam/boot" ]

ENV PATH /opt/conda/envs/env/bin:${PATH}
#  This is necessary so findlibs can find the eccodes library for metview
#  Beam workers do not trigger a default conda environment activation
ENV CONDA_PREFIX /opt/conda/envs/env

# Without following line, the following error occurs on import of pandas:
# ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
RUN cp $CONDA_PREFIX/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/

RUN python -m eccodes selfcheck && python -m cfgrib selfcheck && python -m metview selfcheck
