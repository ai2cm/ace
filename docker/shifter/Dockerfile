# Dockerfile for fme:dev shifter image; doesn't install full-model/fme.

FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV FME_DIR=/full-model
ENV DGLBACKEND=pytorch

# Equivalent to passing --no-cache-dir to every pip install
ENV PIP_NO_CACHE_DIR=off

# install remaining python deps
COPY requirements_except_torch.txt /tmp/requirements.txt
COPY requirements_no_deps.txt /tmp/requirements_no_deps.txt
RUN python3 -m pip install -r /tmp/requirements.txt
RUN python3 -m pip install --no-deps -r /tmp/requirements_no_deps.txt

# copy local code
COPY models/FourCastNet ${FME_DIR}/models/FourCastNet

# install packages in full-model repo, except for fme
RUN cd $FME_DIR && python3 -m pip install --no-deps -e models/FourCastNet