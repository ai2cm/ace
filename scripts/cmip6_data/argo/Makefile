IMAGE = us-central1-docker.pkg.dev/vcm-ml/full-model/cmip6-processing

# Build context is the parent directory (scripts/cmip6_data/) so that
# the Dockerfile can COPY *.py scripts alongside the requirements.
.PHONY: build
build:
	docker build -f Dockerfile -t $(IMAGE):latest ..

.PHONY: push
push:
	docker push $(IMAGE):latest

.PHONY: build_and_push
build_and_push: build push
