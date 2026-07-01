# Makefile for downscaling analysis workflows.

# Conda environment to install into (matches the repo default in the root Makefile).
ENVIRONMENT_NAME ?= fme

.PHONY: tc_deps

# Install TempestExtremes (DetectNodes/StitchNodes) for TC detection.
# TempestExtremes is a compiled binary and is only distributed on conda-forge
# (not pip), so it is installed separately from the pip-based `fme` env. This
# is required by detect_tc_tracks.py.
tc_deps:
	conda install -n $(ENVIRONMENT_NAME) -c conda-forge tempest-extremes
