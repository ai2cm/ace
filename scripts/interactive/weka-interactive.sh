#!/bin/bash

cluster_name=$1
home_dir_subpath=$2
n_gpus=$3
image_override=$4

if [ ! -z "$n_gpus" ]; then
    gpu_arg="--gpus $n_gpus"
else
    gpu_arg=""
fi

if [ "$image_override" = "default" ]; then
    image_flag=""
elif [ ! -z "$image_override" ]; then
    image_flag="--image beaker://${image_override}"
else
    image_flag="--image beaker://jeremym/fme-deps-only-5a6649b8"
fi

beaker session create --remote --bare --cluster ${cluster_name} \
    --budget ai2/climate \
    --workdir /root \
    ${image_flag} \
    --mount src=weka,ref=climate-default,subpath=${home_dir_subpath},dst=/root \
    --mount src=weka,ref=climate-default,dst=/climate-default \
    --mount src=secret,ref=github-ssh-key,dst=/secrets/ghub \
    --mount src=secret,ref=beaker-config,dst=/secrets/.beaker/config.yml \
    --mount src=secret,ref=google-credentials,dst=/secrets/google_app_credentials.json \
    --env GOOGLE_APPLICATION_CREDENTIALS=/secrets/google_app_credentials.json \
    --secret-env WANDB_API_KEY=wandb-api-key \
    ${gpu_arg} \
    bash

