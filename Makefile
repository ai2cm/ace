VERSION ?= $(shell git rev-parse HEAD)
IMAGE ?= fourcastnet

build_docker_image:
	docker build -f docker/Dockerfile -t $(IMAGE):$(VERSION) .

build_beaker_image: build_docker_image
	beaker image create --name $(IMAGE)-$(VERSION) $(IMAGE):$(VERSION)

enter_docker_image: build_docker_image
	docker run -it --rm $(IMAGE):$(VERSION) bash
