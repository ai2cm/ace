#!/bin/bash

BEAKER_IMAGE=$1
DIR=$(pwd)

beaker session create \
	--bare \
	--gpus 1 \
	--image beaker://${BEAKER_IMAGE} \
	--secret-env WANDB_API_KEY=wandb-api-key \
	--shared-memory 10GiB \
	--mount hostpath://${DIR}=/full-model \
	--workdir /full-model/fme
