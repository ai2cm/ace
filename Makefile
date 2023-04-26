VERSION ?= $(shell git rev-parse HEAD)
IMAGE ?= fme
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

