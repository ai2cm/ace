# Scripts for generating a dataset for Full Model Emulation based on ERA5

## Environment

The beam pipeline is run using Dataflow. It first
requires creating a local Python environment with the needed dependencies
installed:

```
make create_environment
```

If needed the Docker image required for running the workflow in the cloud can
be rebuilt and pushed using:

```
make build_dataflow push_dataflow
```

## Computing coarsened ERA5 dataset for FME

Once the local environment has been created and activated, and the docker
image has been pushed, the pipeline can be launched with:

```
make era5_dataflow
```

This will run the production workflow on the full range of data targeting one
degree and four degree grids. During development it can be helpful to run the
workflow for a shorter period of time targeting just a single grid. For this
purpose use the

```
make era5_dataflow_test_run
```

rule as a starting point.
