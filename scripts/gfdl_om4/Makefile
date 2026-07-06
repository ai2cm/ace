LOCAL_ENVIRONMENT ?= gfdl-om4-ingestion

HGRID_URL ?= gs://vcm-ml-scratch/jamesd/2024-11-11-static-data/ocean_hgrid.nc
WEIGHTS_VERSION ?= v1
WEIGHTS_URL_ROOT ?= gs://vcm-ml-scratch/jamesd/gfdl-om4-regridding-weights/$(WEIGHTS_VERSION)

create_environment:
	conda env create -f environment.yaml
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) pip install -r dataflow-requirements.txt

generate_weights_one_degree:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.weights \
		--hgrid-url $(HGRID_URL) \
		--target-grid F90 \
		--output-url $(WEIGHTS_URL_ROOT)/om4-tripolar-0.25deg-to-F90

generate_weights_four_degree:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.weights \
		--hgrid-url $(HGRID_URL) \
		--target-grid F22.5 \
		--output-url $(WEIGHTS_URL_ROOT)/om4-tripolar-0.25deg-to-F22.5

.PHONY: create_environment generate_weights_one_degree generate_weights_four_degree
