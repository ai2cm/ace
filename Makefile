VERSION ?= $(shell git rev-parse --short HEAD)
IMAGE ?= fme
REGISTRY ?= registry.nersc.gov/m4492/ai2cm
ENVIRONMENT_NAME ?= fme
USERNAME ?= $(shell beaker account whoami --format=json | jq -r '.[0].name')
DEPLOY_TARGET ?= pypi

build_docker_image:
	docker build -f docker/Dockerfile -t $(IMAGE):$(VERSION) .

push_shifter_image: build_docker_image
	docker tag $(IMAGE):$(VERSION) $(REGISTRY)/$(IMAGE):$(VERSION)
	docker push $(REGISTRY)/$(IMAGE):$(VERSION)

build_beaker_image: build_docker_image
	beaker image create --name $(IMAGE)-$(VERSION) $(IMAGE):$(VERSION)

build_podman_image:
	podman-hpc build -f docker/Dockerfile -t $(IMAGE):$(VERSION) .

migrate_podman_image: build_podman_image
	podman-hpc migrate $(IMAGE):$(VERSION)

enter_docker_image: build_docker_image
	docker run -it --rm $(IMAGE):$(VERSION) bash

launch_beaker_session:
	./launch-beaker-session.sh $(USERNAME)/$(IMAGE)-$(VERSION)

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
