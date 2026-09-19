VERSION ?= latest
IMAGE_NAME ?= us-central1-docker.pkg.dev/vcm-ml/full-model/glorys-ingest-dataflow:$(VERSION)
LOCAL_ENVIRONMENT ?= glorys-ingest
DATE := $(shell date +%Y-%m-%d)
DATASET_NAME ?= glorys12-ocean-1deg-19level-5day-1993-2025

# Production run parameters.
# GLORYS12 daily means: 1993-01-01 .. 2026-06-23 (reanalysis to 2021-06-30,
# interim after). ERA5 1-degree forcing store ends 2025-12-31T18, and each
# output state needs the 5-day forcing window ENDING at it, so the last
# output state is the last stride-5 day <= 2025-12-31: 2025-12-27.
START_DATE=1993-01-01
END_DATE=2025-12-27
OUTPUT_PATH_ONE_DEGREE=gs://vcm-ml-intermediate/$(DATE)-$(DATASET_NAME).zarr
OUTPUT_GRID_ONE_DEGREE=F90

# Test run parameters (Dataflow): 2 months = 13 states
START_DATE_TEST_RUN ?= 2015-11-01
END_DATE_TEST_RUN ?= 2015-12-31
OUTPUT_PATH_TEST_RUN ?= gs://vcm-ml-scratch/$(DATE)-glorys-test/ocean-1deg.zarr
OUTPUT_GRID_TEST_RUN ?= F90

# Local debug parameters (2 states)
START_DATE_LOCAL ?= 2015-11-01
END_DATE_LOCAL ?= 2015-11-06
OUTPUT_PATH_LOCAL ?= /tmp/glorys-local-test.zarr

create_environment:
	conda env create -f environment.yaml
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) pip install -r dataflow-requirements.txt

build_dataflow:
	docker build --platform=linux/amd64 -t $(IMAGE_NAME) .

push_dataflow: build_dataflow
	docker push $(IMAGE_NAME)

enter:
	docker run --rm -v $$(pwd):/glorys -w /glorys --entrypoint "/bin/bash" -it $(IMAGE_NAME)

# Pass --regrid_weights <gs:// or local path> here to skip the per-worker ESMF
# weight build (~3.5 min and ~10 GB each); build the file with
# make_regrid_weights.py. MAX_NUM_WORKERS overrides the autoscaling cap.
DATAFLOW_EXTRA_FLAGS ?=

.PHONY: glorys_dataflow
glorys_dataflow:
	cd pipeline && \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh DataflowRunner \
		$(OUTPUT_PATH_ONE_DEGREE) $(OUTPUT_GRID_ONE_DEGREE) $(START_DATE) $(END_DATE) \
		$(DATAFLOW_EXTRA_FLAGS)

.PHONY: glorys_dataflow_test_run
glorys_dataflow_test_run:
	cd pipeline && \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh DataflowRunner \
		$(OUTPUT_PATH_TEST_RUN) $(OUTPUT_GRID_TEST_RUN) $(START_DATE_TEST_RUN) $(END_DATE_TEST_RUN) \
		$(DATAFLOW_EXTRA_FLAGS)

# Minimal local test: 2 states, DirectRunner. Generating the 1/12-degree ->
# F90 conservative weights takes minutes and several GB of RAM, and is the
# largest term in peak memory; build them once with make_regrid_weights.py and
# pass EXTRA_FLAGS="--regrid_weights <file>" to reuse them.
LOCAL_EXTRA_FLAGS ?=
.PHONY: glorys_local_debug
glorys_local_debug:
	rm -rf $(OUTPUT_PATH_LOCAL)
	cd pipeline && \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		python glorys-pipeline.py $(OUTPUT_PATH_LOCAL) $(START_DATE_LOCAL) $(END_DATE_LOCAL) \
		--output_grid F90 --runner DirectRunner --save_main_session \
		--job_server_timeout=3600 $(LOCAL_EXTRA_FLAGS)

# Normalization stats (incl. the residual/tendency stats the hybrid ocean uses)
.PHONY: compute_stats
compute_stats:
	cd ../data_process && \
		./compute_stats.sh --config configs/glorys12-ocean-1deg-19level-5day.yaml
