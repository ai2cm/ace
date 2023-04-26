FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV FME_DIR=/full-model
ENV DGLBACKEND=pytorch

# install python deps
COPY requirements_except_torch.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# copy local code
COPY models/FourCastNet ${FME_DIR}/models/FourCastNet
COPY models/fcn-mip ${FME_DIR}/models/fcn-mip
COPY fme ${FME_DIR}/fme
COPY install_local_packages.sh ${FME_DIR}/install_local_packages.sh

# install packages in full-model repo
RUN cd $FME_DIR && ./install_local_packages.sh
