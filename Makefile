MODELS = baseline_afno_26 afno_26ch_v hafno_baseline_26ch_edim512_mlp2 sfno_73ch sfno_73ch_5step graphcast_34ch sfno_73ch_3step sfno_73ch_9step
ACCS = $(addsuffix .nc,$(addprefix 34Vars/acc/, $(MODELS))) 34Vars/acc/ifs.nc
LONG = $(addsuffix .nc,$(addprefix 34Vars/long/, $(MODELS)))

all: 34Vars/acc/tfno_jean.nc 34Vars/acc/baseline_afno_26.nc 34Vars/long/tfno_jean.zarr 34Vars/long/baseline_afno_26.zarr 34Vars/long/baseline_afno_26.zarr \
	34Vars/acc/gfno_26ch_sc3_layers8_tt64.nc

unit_test: ARGS = ""
unit_test:
	ldconfig
	coverage run -a -m pytest -r xfXs $(ARGS)

test: unit_test
	ldconfig
	./runtests.sh

index.html: $(ACCS) $(LONG)
	python3 plots/report.py

report/acc_z500.png: $(ACCS)
	mkdir -p report
	python3 plots/plot_acc.py -o $@ -v z500 $^

34Vars/acc/ifs.nc:
	python3 score-ifs.py --n-jobs 256 $@

34Vars/acc/%.nc:
	torchrun --nproc_per_node 8 -m fcn_mip.inference_medium_range -n 56 $* $@

34Vars/long/%.nc:
	mkdir -p 34Vars/long
	WORLD_SIZE=1 bin/inference_long.sh $* $@

34Vars/2year/%.zarr:
	mkdir -p 34Vars/2year
	python3 bin/inference.py --every 4 --channels u250,v250,tcwv --n-timesteps 2920 $* $@

# docker stuff
SHA = $(shell git rev-parse HEAD)
TAG = 23.04.04
IMAGE = gitlab-master.nvidia.com:5005/earth-2/fcn-mip:$(TAG)
IMAGE_NGC = nvcr.io/nvidian/fcn-mip:$(TAG)

docker_login:
	@docker login -u nbrenowitz -p $(GITLAB_TOKEN) gitlab-master.nvidia.com:5005

docker_run:
	docker run -it --rm --gpus all --net=host --ipc=host -v ${PWD}:/code $(IMAGE) /bin/bash

build: update_submodules
	docker build --platform linux/amd64 --build-arg GIT_COMMIT_SHA=$(SHA) -t $(IMAGE) .

install_local_dependencies:
	pip install external/torch_sht
	pip install external/modulus-core
	pip install -e .

push: build
	docker push $(IMAGE)

pull:
	docker pull $(IMAGE)

push_ngc:
	docker tag $(IMAGE) $(IMAGE_NGC)
	ngc registry image push $(IMAGE_NGC)

setup-hooks:
	git lfs install
	pre-commit install

lint:
	pre-commit run --all-files

update_submodules:
	git submodule update --init external/torch_sht/
	git submodule update --init external/modulus-core/

.PHONY: mount
