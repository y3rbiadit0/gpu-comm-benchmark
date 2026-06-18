#!/usr/bin/env bash
set -euo pipefail

cp_vector_add_setup() {
  : "${CP_STACK:?missing CP_STACK}"
  : "${CP_RUNTIME:?missing CP_RUNTIME}"
  : "${CP_BINARY:?missing CP_BINARY}"
  : "${CP_RESULT_NAME:?missing CP_RESULT_NAME}"
  : "${CP_NODES:?missing CP_NODES}"
  : "${CP_TASKS_PER_NODE:?missing CP_TASKS_PER_NODE}"

  mkdir -p ./logs

  export LC_ALL=C
  export COMM_PLAYGROUND_JOB_NODES="$CP_NODES"

  source "$CP_PROJECT_ROOT/cluster/leonardo/environment.sh" "$CP_STACK"
  source "$CP_PROJECT_ROOT/cluster/leonardo/runtime/$CP_RUNTIME.sh"

  CP_N=${CP_N:-1048576}
  CP_NTRIALS=${CP_NTRIALS:-3}
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-}
  CP_RUN_DIR=${CP_RUN_DIR:-results/$CP_RESULT_NAME/vector_add}

  [ -x "$CP_BINARY" ] || { echo "no executable: $CP_BINARY" >&2; exit 1; }
}

cp_vector_add_print_summary() {
  echo "job: ${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-manual}"
  echo "nodes: ${SLURM_NODELIST:-manual}"
  echo "node: $(hostname)"
  echo "project_root: $CP_PROJECT_ROOT"
  echo "binary: $CP_BINARY"
  echo "result name: $CP_RESULT_NAME"
  echo "problem size: $CP_N"
  echo "trials: $CP_NTRIALS"
  echo "nodes requested: $CP_NODES"
  echo "tasks per node: $CP_TASKS_PER_NODE"
  echo "stack: $CP_STACK"
  echo "runtime: $CP_RUNTIME"
  echo "UCX_TLS: ${UCX_TLS:-unset}"
  echo "CCL_BACKEND: ${CCL_BACKEND:-unset}"
  echo "CCL_ATL_TRANSPORT: ${CCL_ATL_TRANSPORT:-unset}"
  echo "CCL_MPI_LIBRARY_PATH: ${CCL_MPI_LIBRARY_PATH:-unset}"
  echo "CCL_WORKER_COUNT: ${CCL_WORKER_COUNT:-unset}"
  echo "NCCL_DEBUG: ${NCCL_DEBUG:-unset}"
  nvidia-smi || true
}

cp_vector_add_run_trials() {
  for trial in $(seq "$CP_NTRIALS"); do
    local outfile="$CP_RUN_DIR/${SLURM_JOB_NAME:-manual}-${SLURM_JOB_ID:-manual}-${trial}-stdout.txt"
    local errfile="$CP_RUN_DIR/${SLURM_JOB_NAME:-manual}-${SLURM_JOB_ID:-manual}-${trial}-stderr.txt"

    mkdir -p "$(dirname "$outfile")" "$(dirname "$errfile")"

    echo "$CP_RESULT_NAME - Trial $trial of $CP_NTRIALS"
    echo "stdout: ${outfile}.tmp"
    echo "stderr: ${errfile}.tmp"

    /usr/bin/time -p --verbose \
      srun --cpu-freq=high \
      -N "$CP_NODES" \
      --ntasks-per-node="$CP_TASKS_PER_NODE" \
      "$CP_PROJECT_ROOT/cluster/leonardo/gpu-rank-wrapper.sh" \
      "$CP_BINARY" \
      "$CP_N" \
      $CP_EXTRA_ARGS \
      >"${outfile}.tmp" 2>"${errfile}.tmp" \
    && mv --verbose "${outfile}.tmp" "$outfile" \
    && mv --verbose "${errfile}.tmp" "$errfile"
  done
}

cp_vector_add_main() {
  cp_vector_add_setup
  cp_vector_add_print_summary
  cp_vector_add_run_trials
}
