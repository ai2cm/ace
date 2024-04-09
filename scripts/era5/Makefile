VERSION ?= 2024-03-20-netcdf-to-zarr-via-xarray-beam
IMAGE_NAME ?= us.gcr.io/vcm-ml/era5-ingest-dataflow:$(VERSION)
LOCAL_ENVIRONMENT ?= era5-ingestion

ingest_ncar_variables:
	cd ingest_ncar_data && ./main.sh

create_environment:
	conda env create -f environment.yaml
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) pip install -r requirements.txt

build_dataflow:
	docker build -t $(IMAGE_NAME) .

push_dataflow: build_dataflow
	docker push $(IMAGE_NAME)

enter:
	docker run --rm -v $$(pwd):/era5 -w /era5 --entrypoint "/bin/bash" -it $(IMAGE_NAME)

netcdf_to_zarr_local:
	cd netcdf_to_zarr && conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) ./run-netcdf-to-zarr.sh DirectRunner $(IMAGE_NAME)

netcdf_to_zarr_dataflow:
	cd netcdf_to_zarr && conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) ./run-netcdf-to-zarr.sh DataflowRunner $(IMAGE_NAME)
