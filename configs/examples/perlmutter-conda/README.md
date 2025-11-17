# NERSC Perlmutter workflow using a conda environment

This guide details some of the steps to get up and running ACE
training and inference on [NERSC's](https://docs.nersc.gov/) [Perlmutter
system](https://docs.nersc.gov/systems/perlmutter/)!

* [Prerequisites](#prerequisites): Create your NERSC account
* [Preliminary setup](#preliminary-setup): Set up SSH, wandb, and put your data on Perlmutter
* [Run ACE on Perlmutter](#running-ace-on-perlmutter): Submit a Slurm batch job to train ACE on Perlmutter

# Prerequisites

You will need a NERSC account associated with an active project to run jobs on
Perlmutter. Beyond the information below, see the
[docs](https://docs.nersc.gov/accounts/) for more details.

## NERSC account request

Whether you already have a NERSC account or need a new account, you can use the
form at https://iris.nersc.gov/add-user to request access to an existing
allocation. You can reach out to your collaborators for the project name and
help with verbiage for the request.

Note that a valid [ORCID iD](https://orcid.org/) is required when requesting a
new account.

# Preliminary Setup

## SSH into Perlmutter via `sshproxy`

Once you have a NERSC account and [setup
MFA](https://docs.nersc.gov/connect/mfa/#multi-factor-authentication-mfa), you
can use a tool called [`sshproxy`](https://docs.nersc.gov/connect/mfa/#sshproxy)
that NERSC provides to generate an SSH key that is valid for 24 hours.
`sshproxy` generates a private-public key pair that is stored in your local
`~/.ssh` directory, so you can avoid having to do MFA for every connection. Add
Perlmutter to your `~/.ssh/config` file for easy login:

```sh
Host perlmutter
     HostName perlmutter-p1.nersc.gov
     User <nersc-username>
     IdentityFile ~/.ssh/nersc
```

## Preparing your data

Your data should ideally be located on [Perlmutter's Scratch Lustre
filesystem](https://docs.nersc.gov/filesystems/perlmutter-scratch/). Every user
account has a corresponding Scratch directory with a 20TB quota (use `showquota`
to see your home and scratch directory usage). How you get the data to
Perlmutter is up to you, but the recommended way to do large transfers is using
[Perlmutter's dedicated Globus
Endpoint](https://docs.nersc.gov/systems/perlmutter/#transferring-data-to-from-perlmutter-scratch)
which has direct access to Perlmutter Scratch. Another option is to stage the
data on NERSC's [Community File System
(CFS)](https://docs.nersc.gov/filesystems/community/#community-file-system-cfs),
but first this must be setup at the project level by the allocation's PI and
also requires a two-hop transfer with the move from CFS to Scratch.

## Preparing your config

Example configurations can be found in the directory
`configs/examples/perlmutter-conda/config-train.yaml` and
`configs/examples/perlmutter-conda/config-inference.yaml`. Other examples can be found
under the `configs/baselines` directory.

You may choose to reuse the same pattern in the config examples where `FME_TRAIN_DIR`, `FME_VALID_DIR`, and `FME_STATS_DIR` are set through environment variables, or modify them directly to the path to your data.

## Put your WandB API key in `~/.config/wandb/api`

Finally, to log to WandB you'll need to create a file `~/.config/wandb/api`
which contains your API key.

# Running ACE on Perlmutter

`run-train-perlmutter.sh` and `run-inference-perlmutter.sh` are scripts that
will submit the training and standalone inference jobs to Slurm. You will need to
specify the ACE commit hash you want to use and modify the paths to your data.
During training and inference, we first make a conda virtual environment with the
specified commit hash. If the environment already exists, it will be reused. By default,
all results will be saved to an environment variable `FME_OUTPUT_DIR`
which defaults to `${PSCRATCH}/fme-output/<JOBID>`.

If your job fails, you can resume it by setting `RESUME_JOB_ID` to the job ID of the failed job.

Since we used `-J` with `sbatch`, Slurm will create the file
`joblogs/<JOBID>.out` with the run's stdout and stderr, where
`<JOBID>` is the job's Slurm ID. To keep logs from cluttering your working
directory, the default puts them in a subdirectory using
`--output=joblogs/%x-%j.out` (Slurm will handle creating the `joblogs` directory
if it doesn't already exist). You can modify this behavior using the `sbatch`
option `--output`, e.g. `--output=/dev/null` to prevent Slurm from creating a
log file.