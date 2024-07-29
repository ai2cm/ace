VERSION ?= 2024-07-10-era5-xarray-beam-pipelines
IMAGE_NAME ?= us.gcr.io/vcm-ml/era5-ingest-dataflow:$(VERSION)
LOCAL_ENVIRONMENT ?= era5-ingestion

ingest_ncar_variables:
	cd ingest_ncar_data && ./main.sh

create_environment:
	conda env create -f environment.yaml
	conda run --no-capture-output -n era5-ingestion pip install -r dataflow-requirements.txt
	conda run --no-capture-output -n era5-ingestion pip install metview jupyterlab matplotlib cfgrib

build_dataflow:
	docker build -t $(IMAGE_NAME) .

push_dataflow: build_dataflow
	docker push $(IMAGE_NAME)

enter:
	docker run --rm -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it $(IMAGE_NAME)

enter_google:
	docker run --rm -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it gcr.io/weather-tools-prod/weather-tools:0.0.0

netcdf_to_zarr_local:
	cd netcdf_to_zarr && conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) ./run-netcdf-to-zarr.sh DirectRunner $(IMAGE_NAME)

netcdf_to_zarr_dataflow:
	cd netcdf_to_zarr && conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) ./run-netcdf-to-zarr.sh DataflowRunner $(IMAGE_NAME)

era5_dataflow:
	cd pipeline && conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) ./run-dataflow.sh
