## Examples

Example run scripts for using fcn-mip utilities.

- [simple.py](./simple.py): run an SFNO model for 40 timesteps and save plots
  from the TCWV field as a png

### Basic inference

Pull the data

    git lfs pull

To run a basic inference with the SFNO 73 channel model run from the root directory:

    torchrun --nproc_per_node 1 bin/inference_ensemble.py \
        examples/config.json
