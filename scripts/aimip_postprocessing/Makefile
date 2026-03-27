ENVIRONMENT_NAME = fme
LOCAL_DIR = /tmp/aimip-ace/

create_environment:
	conda run -n base mamba install -y -n $(ENVIRONMENT_NAME) \
		fsspec gcsfs s3fs xarray cftime python-dotenv click pyyaml netcdf4

postprocess:
	conda run -n $(ENVIRONMENT_NAME) python postprocess.py

test:
	conda run -n $(ENVIRONMENT_NAME) python -m pytest test_postprocess.py -v --noconftest

clean:
	rm -rf $(LOCAL_DIR)

.PHONY: create_environment postprocess test clean
