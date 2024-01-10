VERSION ?= $(shell git rev-parse --short HEAD)
IMAGE ?= fme
REGISTRY ?= registry.nersc.gov/m4331/ai2cm
ENVIRONMENT_NAME ?= fme
USERNAME ?= $(shell beaker account whoami --format=json | jq -r '.[0].name')

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

install_local_packages:
	./install_local_packages.sh

install_dependencies:
	./install_dependencies.sh

# recommended to deactivate current conda environment before running this
create_environment:
	conda create -n $(ENVIRONMENT_NAME) python=3.10 pip
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) ./install_dependencies.sh
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) ./install_local_packages.sh

test_fme_unit_tests:
	pytest -m "not requires_gpu" --durations 10 fme/
