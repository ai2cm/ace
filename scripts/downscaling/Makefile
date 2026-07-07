# Makefile for downscaling analysis workflows.

# Conda environment to install into (matches the repo default in the root Makefile).
ENVIRONMENT_NAME ?= fme

# Where the TempestExtremes source is cloned and built.
TC_SRC_DIR ?= $(HOME)/.cache/tempestextremes-src
# Prefix TempestExtremes binaries are installed to.
TC_INSTALL_PREFIX ?= $(HOME)/.local/tempestextremes
# Root of the apt-installed NetCDF/HDF5 dev libraries. The conda toolchain
# used to compile (from the active conda env) does not search system
# /usr/include or /usr/lib by default, so this must be passed explicitly.
NETCDF_ROOT ?= /usr

.PHONY: tc_deps

# Build and install TempestExtremes (DetectNodes/StitchNodes) from source via
# CMake. This avoids a conda solve entirely, at the cost of needing NetCDF-C++
# and HDF5 development headers on the system already (e.g. on Debian/Ubuntu:
# `apt install libnetcdf-dev libnetcdf-c++4-dev libhdf5-dev`), plus a C++
# compiler and cmake. This is required by detect_tc_tracks.py; point
# --detect-exe/--stitch-exe at $(TC_INSTALL_PREFIX)/bin/DetectNodes and
# $(TC_INSTALL_PREFIX)/bin/StitchNodes when running the script.
tc_deps:
	if [ ! -d $(TC_SRC_DIR) ]; then \
		git clone https://github.com/ClimateGlobalChange/tempestextremes $(TC_SRC_DIR); \
	fi
	cmake -S $(TC_SRC_DIR) -B $(TC_SRC_DIR)/build \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$(TC_INSTALL_PREFIX) \
		-DNetCDF_ROOT=$(NETCDF_ROOT) \
		-DHDF5_ROOT=$(NETCDF_ROOT)
	$(MAKE) -C $(TC_SRC_DIR)/build
	$(MAKE) -C $(TC_SRC_DIR)/build install
