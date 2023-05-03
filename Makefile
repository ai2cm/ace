VERSION ?= $(shell git rev-parse HEAD)
IMAGE ?= fme
ENVIRONMENT_NAME ?= fme
USERNAME ?= $(shell beaker account whoami --format=json | jq -r '.[0].name')

build_docker_image:
	docker build -f docker/Dockerfile -t $(IMAGE):$(VERSION) .

build_beaker_image: build_docker_image
	beaker image create --name $(IMAGE)-$(VERSION) $(IMAGE):$(VERSION)

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
	conda create -n $(ENVIRONMENT_NAME) python=3.8 pip
	conda run -n $(ENVIRONMENT_NAME) ./install_dependencies.sh
	conda run -n $(ENVIRONMENT_NAME) ./install_local_packages.sh

test_fme_unit_tests:
	pytest -m "not requires_gpu" fme/
