VERSION ?= $(shell git rev-parse --short HEAD)
IMAGE ?= fme
ENVIRONMENT_NAME ?= fme
DEPLOY_TARGET ?= pypi

build_docker_image:
	docker build --platform=linux/amd64 -f docker/Dockerfile -t $(IMAGE):$(VERSION) .

enter_docker_image: build_docker_image
	docker run -it --rm $(IMAGE):$(VERSION) bash

# recommended to deactivate current conda environment before running this
create_environment:
	conda create -n $(ENVIRONMENT_NAME) python=3.10 pip
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) python -m pip install uv
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) uv pip install -c constraints.txt -e ./fme[dev,docs]
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) uv pip install -r analysis-deps.txt

test:
	pytest --durations 40 .

test_fast:
	pytest --durations 40 --fast .

test_very_fast:
	pytest --durations 40 --very-fast .

# For maintainer use only
# requires fme[deploy] to be installed

build_pypi:
	rm -rf fme/dist
	cd fme && python -m build

deploy_pypi: build_pypi
	cd fme && twine upload --repository $(DEPLOY_TARGET) dist/*

deploy_test_pypi: DEPLOY_TARGET = testpypi
deploy_test_pypi: deploy_pypi
