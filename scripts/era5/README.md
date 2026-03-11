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
