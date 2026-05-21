VERSION ?= latest
IMAGE_NAME ?= us-central1-docker.pkg.dev/vcm-ml/full-model/ufs-replay-ingest-dataflow:$(VERSION)
LOCAL_ENVIRONMENT ?= ufs-replay-ingest

# Production run parameters
# Ocean (MOM6): 1993-12-31 18:00 to 2023-12-31 00:00
# Atmo  (FV3):  1993-12-31 18:00 to 2023-10-13 03:00
# Use the overlap: 1994-01-01 to 2023-10-12 (atmo is the binding constraint)
START_TIME=1994-01-01T06:00:00
END_TIME=2023-10-12T18:00:00
OUTPUT_PATH_ONE_DEGREE=gs://vcm-ml-intermediate/ufs-replay-ocean-1deg-19level-1994-2023.zarr
OUTPUT_GRID_ONE_DEGREE=F90

# Test run parameters (Dataflow)
START_TIME_TEST_RUN ?= 2023-10-01T06:00:00
END_TIME_TEST_RUN ?= 2023-10-12T18:00:00
OUTPUT_PATH_TEST_RUN ?= gs://vcm-ml-scratch/ufs-replay-test/ocean-1deg.zarr
OUTPUT_GRID_TEST_RUN ?= F90

# Local debug parameters (just 1 day = 4 ocean timesteps → 1 daily output)
START_TIME_LOCAL ?= 2023-10-01T06:00:00
END_TIME_LOCAL ?= 2023-10-02T00:00:00
OUTPUT_PATH_LOCAL ?= /tmp/ufs-replay-local-test.zarr

create_environment:
	conda env create -f environment.yaml
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) pip install -r dataflow-requirements.txt
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) pip install jupyterlab matplotlib

build_dataflow:
	docker build --platform=linux/amd64 -t $(IMAGE_NAME) .

push_dataflow: build_dataflow
	docker push $(IMAGE_NAME)

enter:
	docker run --rm -v $$(pwd):/ufs-replay -w /ufs-replay --entrypoint "/bin/bash" -it $(IMAGE_NAME)

.PHONY: ufs_replay_dataflow
ufs_replay_dataflow: ufs_replay_dataflow_one_degree

.PHONY: ufs_replay_dataflow_one_degree
ufs_replay_dataflow_one_degree:
	cd pipeline && \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh \
		DataflowRunner \
		$(OUTPUT_PATH_ONE_DEGREE) \
		$(OUTPUT_GRID_ONE_DEGREE) \
		$(START_TIME) \
		$(END_TIME)

.PHONY: ufs_replay_dataflow_test_run
ufs_replay_dataflow_test_run:
	cd pipeline && \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh \
		DataflowRunner \
		$(OUTPUT_PATH_TEST_RUN) \
		$(OUTPUT_GRID_TEST_RUN) \
		$(START_TIME_TEST_RUN) \
		$(END_TIME_TEST_RUN)

# Minimal local test: 1 day of data, runs on a laptop with DirectRunner
.PHONY: ufs_replay_local_debug
ufs_replay_local_debug:
	rm -rf $(OUTPUT_PATH_LOCAL)
	cd pipeline && \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		python xr-beam-pipeline.py \
		$(OUTPUT_PATH_LOCAL) \
		$(START_TIME_LOCAL) \
		$(END_TIME_LOCAL) \
		--output_grid F90 \
		--runner DirectRunner \
		--save_main_session \
		--job_server_timeout=3600
