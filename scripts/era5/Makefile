VERSION ?= 2026-03-03-era5-xarray-beam-pipelines
IMAGE_NAME ?= us-central1-docker.pkg.dev/vcm-ml/full-model/era5-ingest-dataflow:$(VERSION)
LOCAL_ENVIRONMENT ?= era5-ingestion

# Production run parameters
START_TIME=1940-01-01T12:00:00
END_TIME=2025-12-31T18:00:00
OUTPUT_PATH_ONE_DEGREE=gs://vcm-ml-intermediate/2026-03-19-era5-1deg-8layer-1940-2025.zarr
OUTPUT_GRID_ONE_DEGREE=F90
OUTPUT_PATH_FOUR_DEGREE=gs://vcm-ml-intermediate/2026-03-19-era5-4deg-8layer-1940-2025.zarr
OUTPUT_GRID_FOUR_DEGREE=F22.5

# Pressure level run parameters
PRESSURE_LEVEL_START_TIME=1940-01-01T12:00:00
PRESSURE_LEVEL_END_TIME=2025-12-31T18:00:00
PRESSURE_LEVEL_OUTPUT_PATH=gs://vcm-ml-intermediate/era5-pressure-levels-1deg-1940-2025.zarr

# Test run parameters
START_TIME_TEST_RUN ?= 2025-12-30T00:00:00
END_TIME_TEST_RUN ?= 2025-12-31T18:00:00
OUTPUT_PATH_TEST_RUN ?= gs://vcm-ml-scratch/spencerc/test-updated-era5-pipeline/era5-4deg-8layer.zarr
OUTPUT_GRID_TEST_RUN ?= F22.5

create_environment:
	conda env create -f environment.yaml
	conda run --no-capture-output -n era5-ingestion pip install -r dataflow-requirements.txt
	conda run --no-capture-output -n era5-ingestion pip install jupyterlab matplotlib

build_dataflow:
	docker build --platform=linux/amd64 -t $(IMAGE_NAME) .

push_dataflow: build_dataflow
	docker push $(IMAGE_NAME)

enter:
	docker run --rm -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it $(IMAGE_NAME)

enter_google:
	docker run --rm -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it gcr.io/weather-tools-prod/weather-tools:0.0.0

.PHONY: era5_dataflow
era5_dataflow: era5_dataflow_one_degree era5_dataflow_four_degree

.PHONY: era5_dataflow_one_degree
era5_dataflow_one_degree:
	cd pipeline && \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh \
		DataflowRunner \
		$(OUTPUT_PATH_ONE_DEGREE) \
		$(OUTPUT_GRID_ONE_DEGREE) \
		$(START_TIME) \
		$(END_TIME) \
		--check_data_validity

.PHONY: era5_dataflow_four_degree
era5_dataflow_four_degree:
	cd pipeline && \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh \
		DataflowRunner \
		$(OUTPUT_PATH_FOUR_DEGREE) \
		$(OUTPUT_GRID_FOUR_DEGREE) \
		$(START_TIME) \
		$(END_TIME) \
		--check_data_validity

.PHONY: era5_pressure_level_dataflow
era5_pressure_level_dataflow:
	cd pipeline && \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-pressure-level-pipeline.sh \
		DataflowRunner \
		$(PRESSURE_LEVEL_OUTPUT_PATH) \
		$(OUTPUT_GRID_ONE_DEGREE) \
		$(PRESSURE_LEVEL_START_TIME) \
		$(PRESSURE_LEVEL_END_TIME)

.PHONY: era5_dataflow_test_run
era5_dataflow_test_run:
	cd pipeline && \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh \
		DataflowRunner \
		$(OUTPUT_PATH_TEST_RUN) \
		$(OUTPUT_GRID_TEST_RUN) \
		$(START_TIME_TEST_RUN) \
		$(END_TIME_TEST_RUN) \
		--check_data_validity
