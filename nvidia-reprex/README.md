# Extract data

``` bash
tar -xzvf data.tar.gz -C ./
```

# Run training

``` bash
python -m fme.ace.train config.yaml
```

# Config considerations

The example `config.yaml` assumes data is at `./data` and `./results` is a
writeable path.

Batch sizes in `config.yaml` are set with training on a single 80GB A100 in mind.
