# CM4 piControl baseline

## Using the scripts in this directory



## `uncoupled`: Uncoupled atmosphere & ocean pretraining

Uncoupled pretraining and fine-tuning is run in the `uncoupled` directory.

``` sh
cd uncoupled

# train the ocean emulator
bash train.sh ocean/

# train the atmosphere emulator
bash train.sh atmosphere/

# fine-tune the ocean emulator
bash finetune.sh ocean/
```

Training runs are configured in files named `training.txt`, found in the `uncoupled/ocean` and `uncoupled/atmos`. Uncoupled fine-tuning runs are configured in files named `finetuning.txt`.

## `coupled`: Coupled atmosphere-ocean fine-tuning

Coupled fine-tuning is done in the `coupled` directory.

``` sh
cd coupled

# train a new coupled emulator, initialized from pretrained uncoupled components
bash train.sh fto/

# fine-tune a previously trained coupled emulator
bash finetune.sh fto/
```

Training runs are configured in files named `pretraining.txt`; see the example in the `coupled/fto` run directory. To configure a fine-tuning run use `finetuning.txt`.

## Evaluation

Evaluator runs use a common script `evaluate.sh`, found in this directory.

```sh
# run uncoupled atmosphere eval
bash evaluate.sh uncoupled/atmosphere/

# run coupled fto eval
bash evaluate.sh coupled/fto/
```

Evaluation runs for each setting are configured in `experiment.txt` files, e.g.
`coupled/fto/experiments.txt`. The `experiment.txt` file for a specific setting
is automatically updated when a `train.sh` or `finetune.sh` run is submitted.
