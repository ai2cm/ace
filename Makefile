VERSION ?= $(shell git rev-parse --short HEAD)
IMAGE ?= fme
ENVIRONMENT_NAME ?= fme
DEPLOY_TARGET ?= pypi

ifeq ($(shell uname), Linux)
	CONDA_PACKAGES=gxx_linux-64 pip
else
	CONDA_PACKAGES=pip
endif

build_docker_image:
	DOCKER_BUILDKIT=1 docker build --platform=linux/amd64 -f docker/Dockerfile -t $(IMAGE):$(VERSION) --target production .

enter_docker_image: build_docker_image
	docker run -it --rm $(IMAGE):$(VERSION) bash


# recommended to deactivate current conda environment before running this
create_environment:
	conda create -n $(ENVIRONMENT_NAME) python=3.11 $(CONDA_PACKAGES)
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) python -m pip install uv
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) uv pip install -c constraints.txt -e .[dev,docs]
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) uv pip install -r analysis-deps.txt

test:
	pytest --durations 40 .

# --cov must come  after pytest args to use the sources defined by config
test_cov:
	pytest --durations 40 --cov --cov-report=term-missing:skip-covered --cov-config=pyproject.toml .

test_fast:
	pytest --durations 40 --fast .

test_very_fast:
	pytest --durations 40 --very-fast .

# For maintainer use only
# requires fme[deploy] to be installed

build_pypi:
	rm -rf dist
	python -m build

deploy_pypi: build_pypi
	twine upload --repository $(DEPLOY_TARGET) dist/*

deploy_test_pypi: DEPLOY_TARGET = testpypi
deploy_test_pypi: deploy_pypi
