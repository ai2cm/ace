# Gantry configuration

This directory contains configurations and scripts for submitting batch jobs
to Ai2's [beaker](https://beaker.allen.ai/) platform using the
[gantry](https://github.com/allenai/beaker-gantry) tool.

For documentation on beaker, see [here](https://beaker-docs.apps.allenai.org/).
For help with gantry, see the readme in the
[gantry repository](https://github.com/allenai/beaker-gantry) or call `gantry run --help`.

A few important notes about using these examples:
- gantry will install the current commit of this repo before running your job.
Therefore, you must commit and push changes to remote before submitting.
- you can use `--allow-dirty` to skip committing changes, but this may lead
to unexpected behavior so be careful!

For quick experimentation, feel free to make changes to the run scripts or
yaml configurations and commit these changes to experimental branches. These
can be committed to main if they are in the `experiments` directory.
Configurations in `baselines` should reflect the latest best-performing
configuration.
