#!/bin/bash -l

# Mandatory at ALCF: project, walltime, filesystems.
#PBS -A E3SMinput
#PBS -N train-fme
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=04:00:00
#PBS -q preemptable
#PBS -k doe
#PBS -j oe
#PBS -o joblogs/
# Make the job rerunnable so PBS automatically requeues it (with the SAME job id)
# if it gets preempted in the preemptable queue. ACE then resumes from the latest
# checkpoint in experiment_dir.
#PBS -r y

set -xe

cd ${PBS_O_WORKDIR}

# Numeric job id, stable across PBS reruns/preemption (analogous to SLURM_JOB_ID).
JOBID=${PBS_JOBID%%.*}

FME_SCRATCH=${FME_SCRATCH:-/eagle/E3SMinput/elynnwu}

# The id that names the output directory. For a fresh job this is the current
# job id; for a resume (manual RESUME_JOB_ID, or the soft-timer auto-resubmit
# below) it is the original job's id so the run continues in the same directory.
EFFECTIVE_JOB_ID=${RESUME_JOB_ID:-$JOBID}

# directory for saving output from the training job
export FME_OUTPUT_DIR=${FME_SCRATCH}/fme-output/${EFFECTIVE_JOB_ID}
mkdir -p $FME_OUTPUT_DIR

module use /soft/modulefiles
module load conda
conda activate $FME_VENV

# env variables
export WANDB_JOB_TYPE=training
export WANDB_NOTES="Polaris: ${COMMIT}, results: ${FME_OUTPUT_DIR}"
set +x  # don't print API key to logs
# Use WANDB_API_KEY from the environment (e.g. exported in ~/.bashrc) or fall
# back to `wandb login` credentials in ~/.netrc. Only read the legacy key file
# if it exists, and never clobber an already-set key.
if [ -z "${WANDB_API_KEY}" ] && [ -f ~/.config/wandb/api ]; then
  export WANDB_API_KEY=$(cat ~/.config/wandb/api)
fi
set -x

TRAIN_CONFIG=${CONFIG_DIR}/train-config.yaml

# replace placeholders in config with actual values
sed -i "s|FME_OUTPUT_DIR|${FME_OUTPUT_DIR}|" ${TRAIN_CONFIG}
sed -i "s|FME_TRAIN_DIR|${FME_TRAIN_DIR}|" ${TRAIN_CONFIG}
sed -i "s|FME_VALID_DIR|${FME_VALID_DIR}|" ${TRAIN_CONFIG}
sed -i "s|FME_STATS_DIR|${FME_STATS_DIR}|" ${TRAIN_CONFIG}

cp -r $CONFIG_DIR $FME_OUTPUT_DIR/job_config

# ---- distributed (torchrun) setup ----
# Each Polaris compute node has 4x NVIDIA A100 GPUs.
NNODES=$(wc -l < $PBS_NODEFILE)
NGPUS_PER_NODE=4
export NNODES NGPUS_PER_NODE JOBID
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29507
echo "NNODES=${NNODES} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

# Cap per-process thread pools. With "--cpu-bind none" every process can see all
# 64 hardware threads, so without these caps each of the many torchrun + forkserver
# dataloader workers would spawn a 64-thread BLAS/OpenMP pool and exhaust the
# node's thread limit (OpenBLAS: pthread_create ... Resource temporarily unavailable).
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# zarr v3 decodes chunks with an async thread pool whose size derives from
# os.cpu_count() (=64; taskset affinity does NOT change os.cpu_count). With
# several dataloader workers per rank, plus the validation/inference forkserver
# workers at the epoch boundary, this exhausts the node's thread/pid limit
# ("RuntimeError: can't start new thread"). Cap zarr's async concurrency and the
# torch inductor compile worker pool to keep the total thread count bounded.
export ZARR_ASYNC__CONCURRENCY=4
export TORCHINDUCTOR_COMPILE_THREADS=1

# ---- soft timer: checkpoint and resubmit before the wall clock ----
# PBS (unlike SLURM) has no "send a signal N seconds before walltime" option and
# does NOT auto-requeue on walltime expiry. So we arm our own timer a margin
# before the requested walltime; when it fires we signal training to stop
# (it resumes from its latest checkpoint) and resubmit this job with
# RESUME_JOB_ID set. Override the margin (seconds) by exporting SOFT_LIMIT_MARGIN.
SOFT_LIMIT_MARGIN=${SOFT_LIMIT_MARGIN:-300}

# Resource settings reused when resubmitting. These are forwarded via "qsub -V"
# from run-train-polaris.sh; fall back to sensible defaults / the actual
# allocation if pbs-train.sh was submitted directly.
PROJECT=${PROJECT:-E3SMinput}
QUEUE=${QUEUE:-preemptable}
NUM_NODES=${NUM_NODES:-${NNODES}}
FILESYSTEMS=${FILESYSTEMS:-home:eagle}

# Determine the requested walltime (prefer the forwarded WALLTIME, else ask PBS).
if [ -z "${WALLTIME}" ]; then
  WALLTIME=$(qstat -f "${PBS_JOBID}" 2>/dev/null \
    | awk -F'= ' '/Resource_List.walltime/ {gsub(/ /,"",$2); print $2}')
fi
walltime_seconds() {
  local wt="$1" h m s
  [ -z "$wt" ] && { echo ""; return; }
  IFS=: read -r h m s <<< "$wt"
  if [ -z "$s" ]; then s="$m"; m="$h"; h=0; fi
  echo $(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
WALL_SECONDS=$(walltime_seconds "${WALLTIME}")

TIMER_SENTINEL=${FME_OUTPUT_DIR}/.soft_timer_fired.${JOBID}
rm -f "${TIMER_SENTINEL}"

# Informational: which PBS run/rerun of this job id this is (preemption reruns
# increment run_count). Rendezvous itself is static (see requeueable-train.sh).
RUN_COUNT=$(qstat -f "${PBS_JOBID}" 2>/dev/null | awk '/run_count/ {print $3}')
echo "PBS run_count=${RUN_COUNT:-0} for job ${JOBID}"

# Launch one rank per node via PALS; each rank starts a torchrun agent that
# spawns NGPUS_PER_NODE workers. The agents use static rendezvous (fixed
# node rank + MASTER_ADDR:MASTER_PORT) to form a single
# (NNODES x NGPUS_PER_NODE) process group; see requeueable-train.sh.
mpiexec -n ${NNODES} --ppn 1 \
  --cpu-bind none \
  --env CONFIG_DIR=${CONFIG_DIR} \
  --env FME_VENV=${FME_VENV} \
  --env MASTER_ADDR=${MASTER_ADDR} \
  --env MASTER_PORT=${MASTER_PORT} \
  --env NNODES=${NNODES} \
  --env NGPUS_PER_NODE=${NGPUS_PER_NODE} \
  --env JOBID=${JOBID} \
  --env OUTPUT_JOB_ID=${EFFECTIVE_JOB_ID} \
  --env OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  --env OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} \
  --env MKL_NUM_THREADS=${MKL_NUM_THREADS} \
  --env ZARR_ASYNC__CONCURRENCY=${ZARR_ASYNC__CONCURRENCY} \
  --env TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS} \
  ${CONFIG_DIR}/requeueable-train.sh &
TRAIN_PID=$!

# Arm the watchdog only if we know the walltime and it exceeds the margin.
WATCHDOG_PID=""
if [ -n "${WALL_SECONDS}" ] && [ "${WALL_SECONDS}" -gt "${SOFT_LIMIT_MARGIN}" ]; then
  SOFT_LIMIT=$(( WALL_SECONDS - SOFT_LIMIT_MARGIN ))
  echo "Soft timer armed: checkpoint+resubmit after ${SOFT_LIMIT}s (walltime ${WALLTIME} minus ${SOFT_LIMIT_MARGIN}s margin)."
  (
    sleep ${SOFT_LIMIT}
    echo "Soft time limit reached; signaling training to stop for resubmission."
    touch "${TIMER_SENTINEL}"
    kill -TERM ${TRAIN_PID} 2>/dev/null
  ) &
  WATCHDOG_PID=$!
else
  echo "WARNING: could not determine a usable walltime; soft-timer auto-resubmit is DISABLED for this job."
fi

# Wait for training without letting 'set -e' abort on a non-zero exit.
TRAIN_RC=0
wait ${TRAIN_PID} || TRAIN_RC=$?

# Training returned: cancel the watchdog if it is still sleeping.
if [ -n "${WATCHDOG_PID}" ]; then
  kill ${WATCHDOG_PID} 2>/dev/null || true
  wait ${WATCHDOG_PID} 2>/dev/null || true
fi

if [ -f "${TIMER_SENTINEL}" ]; then
  rm -f "${TIMER_SENTINEL}"
  echo "Soft timer fired -> resubmitting to continue ${EFFECTIVE_JOB_ID} from its latest checkpoint."
  export RESUME_JOB_ID=${EFFECTIVE_JOB_ID}
  qsub -V \
    -A ${PROJECT} \
    -q ${QUEUE} \
    -l select=${NUM_NODES}:system=polaris \
    -l place=scatter \
    -l filesystems=${FILESYSTEMS} \
    -l walltime=${WALLTIME} \
    ${PBS_O_WORKDIR}/pbs-scripts/pbs-train.sh
elif [ ${TRAIN_RC} -eq 0 ]; then
  echo "Training completed successfully."
else
  echo "Training exited with code ${TRAIN_RC} before the soft limit; not resubmitting (preemption is handled by '#PBS -r y')."
fi

sleep 120
