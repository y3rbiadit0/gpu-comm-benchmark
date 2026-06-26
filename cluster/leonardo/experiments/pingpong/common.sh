#!/usr/bin/env bash
set -euo pipefail

cp_pingpong_setup() {
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

  # pingpong is a 2-endpoint benchmark. By default it sweeps powers of two from
  # 1 element up to CP_N; CP_MSG_SIZES can override that with a comma-separated list.
  CP_N=${CP_N:-4194304}
  CP_ITERS=${CP_ITERS:-100}
  CP_WARMUP=${CP_WARMUP:-20}
  CP_NTRIALS=${CP_NTRIALS:-3}
  # pingpong binaries accept: <max_elements> [iterations] [warmup] [message_sizes]
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-"$CP_ITERS $CP_WARMUP"}
  if [[ -n "${CP_MSG_SIZES:-}" ]]; then
    CP_EXTRA_ARGS="$CP_EXTRA_ARGS $CP_MSG_SIZES"
  fi
  CP_LAUNCHER=${CP_LAUNCHER:-srun}
  CP_RUN_DIR=${CP_RUN_DIR:-results/$CP_RESULT_NAME/pingpong}

  [ -x "$CP_BINARY" ] || { echo "no executable: $CP_BINARY" >&2; exit 1; }
}

cp_pingpong_print_summary() {
  echo "job: ${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-manual}"
  echo "nodes: ${SLURM_NODELIST:-manual}"
  echo "node: $(hostname)"
  echo "project_root: $CP_PROJECT_ROOT"
  echo "binary: $CP_BINARY"
  echo "result name: $CP_RESULT_NAME"
  echo "max message elements: $CP_N"
  echo "message sizes: ${CP_MSG_SIZES:-powers-of-two}"
  echo "iterations: $CP_ITERS"
  echo "warmup: $CP_WARMUP"
  echo "extra args: $CP_EXTRA_ARGS"
  echo "trials: $CP_NTRIALS"
  echo "nodes requested: $CP_NODES"
  echo "tasks per node: $CP_TASKS_PER_NODE"
  echo "launcher: $CP_LAUNCHER"
  echo "launcher path: $(command -v "$CP_LAUNCHER" 2>/dev/null || true)"
  echo "stack: $CP_STACK"
  echo "runtime: $CP_RUNTIME"
  echo "UCX_TLS: ${UCX_TLS:-unset}"
  echo "CCL_BACKEND: ${CCL_BACKEND:-unset}"
  echo "CCL_ATL_TRANSPORT: ${CCL_ATL_TRANSPORT:-unset}"
  echo "CCL_MPI_LIBRARY_PATH: ${CCL_MPI_LIBRARY_PATH:-unset}"
  echo "I_MPI_FABRICS: ${I_MPI_FABRICS:-unset}"
  echo "FI_PROVIDER: ${FI_PROVIDER:-unset}"
  echo "NCCL_DEBUG: ${NCCL_DEBUG:-unset}"
  nvidia-smi || true
}

cp_pingpong_run_trials() {
  for trial in $(seq "$CP_NTRIALS"); do
    local outfile="$CP_RUN_DIR/${SLURM_JOB_NAME:-manual}-${SLURM_JOB_ID:-manual}-${trial}-stdout.txt"
    local errfile="$CP_RUN_DIR/${SLURM_JOB_NAME:-manual}-${SLURM_JOB_ID:-manual}-${trial}-stderr.txt"

    mkdir -p "$(dirname "$outfile")" "$(dirname "$errfile")"

    echo "$CP_RESULT_NAME - Trial $trial of $CP_NTRIALS"
    echo "stdout: ${outfile}.tmp"
    echo "stderr: ${errfile}.tmp"

    if [[ "$CP_LAUNCHER" == "mpirun" ]]; then
      /usr/bin/time -p --verbose \
        mpirun -np "$((CP_NODES * CP_TASKS_PER_NODE))" \
        "$CP_PROJECT_ROOT/cluster/leonardo/gpu-rank-wrapper.sh" \
        "$CP_BINARY" \
        "$CP_N" \
        $CP_EXTRA_ARGS \
        >"${outfile}.tmp" 2>"${errfile}.tmp" \
      && mv --verbose "${outfile}.tmp" "$outfile" \
      && mv --verbose "${errfile}.tmp" "$errfile"
    else
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
    fi
  done
}

cp_pingpong_main() {
  cp_pingpong_setup
  cp_pingpong_print_summary
  cp_pingpong_run_trials
}
