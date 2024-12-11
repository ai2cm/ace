FROM nvcr.io/nvidia/pytorch:23.08-py3

# Clone and install earth2grid if not already installed
RUN PACKAGE=earth2grid && \
    if ! pip show "$PACKAGE" &>/dev/null; then \
        git clone https://github.com/NVlabs/earth2grid.git && \
        cd earth2grid && \
        pip install --no-build-isolation . && \
        cd .. && \
        rm -rf earth2grid; \
    fi