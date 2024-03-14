FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV FME_DIR=/full-model
ENV DGLBACKEND=pytorch

# install python deps
COPY requirements_except_torch.txt /tmp/requirements.txt
COPY requirements_no_deps.txt /tmp/requirements_no_deps.txt
RUN python3 -m pip install -r /tmp/requirements.txt
RUN python3 -m pip install --no-deps -r /tmp/requirements_no_deps.txt

# copy local code
COPY fme ${FME_DIR}/fme
COPY install_local_packages.sh ${FME_DIR}/install_local_packages.sh

# install packages in full-model repo
RUN cd $FME_DIR && ./install_local_packages.sh

# copy after install so editing scripts does not trigger reinstall
COPY scripts ${FME_DIR}/scripts
