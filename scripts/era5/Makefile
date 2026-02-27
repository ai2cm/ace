VERSION ?= 2026-02-24-era5-xarray-beam-pipelines
IMAGE_NAME ?= us-central1-docker.pkg.dev/vcm-ml/full-model/era5-ingest-dataflow:$(VERSION)
LOCAL_ENVIRONMENT ?= era5-ingestion

create_environment:
	conda env create -f environment.yaml
	conda run --no-capture-output -n era5-ingestion pip install -r dataflow-requirements.txt
	conda run --no-capture-output -n era5-ingestion pip install metview jupyterlab matplotlib cfgrib

build_dataflow:
	docker build --platform=linux/amd64 -t $(IMAGE_NAME) .

push_dataflow: build_dataflow
	docker push $(IMAGE_NAME)

enter:
	docker run --rm -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it $(IMAGE_NAME)

enter_google:
	docker run --rm -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it gcr.io/weather-tools-prod/weather-tools:0.0.0

era5_dataflow:
	cd pipeline && conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) ./run-dataflow.sh
