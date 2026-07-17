IMAGE=us-central1-docker.pkg.dev/vcm-ml/full-model/era5-nudging-data
IMAGE_VERSION=v2026.07.0  # Use vYYYY.MM.MICRO (MICRO is zero-indexed)

.PHONY: build_image
build_image:
	docker build -f Dockerfile -t $(IMAGE):$(IMAGE_VERSION) .

.PHONY: push_image
push_image:
	docker push $(IMAGE):$(IMAGE_VERSION)
