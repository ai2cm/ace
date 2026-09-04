DOCKER_IMAGE := spencerc/rclone-gantry
DOCKER_PLATFORM := linux/amd64
BEAKER_WORKSPACE := ai2/ace
BEAKER_IMAGE_NAME := rclone-gantry

.PHONY: build_image push_image

build_image:
	docker build --platform $(DOCKER_PLATFORM) -t $(DOCKER_IMAGE) .

push_image: build_image
	beaker image create $(DOCKER_IMAGE) \
		--name $(BEAKER_IMAGE_NAME) \
		--workspace $(BEAKER_WORKSPACE)
