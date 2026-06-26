#!/usr/bin/env bash
set -euo pipefail

cp_cg_step_setup() {
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

  # cg_step works on a square CP_N x CP_N grid; CP_N is the side length.
  # Kept small so the halo exchange + two reductions dominate over stencil compute.
  CP_N=${CP_N:-512}
  CP_ITERS=${CP_ITERS:-50}
  CP_WARMUP=${CP_WARMUP:-10}
  CP_NTRIALS=${CP_NTRIALS:-3}
  # cg_step binaries accept: <side> [iterations] [warmup]
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-"$CP_ITERS $CP_WARMUP"}
  CP_LAUNCHER=${CP_LAUNCHER:-srun}
  CP_RUN_DIR=${CP_RUN_DIR:-results/$CP_RESULT_NAME/cg_step}

  [ -x "$CP_BINARY" ] || { echo "no executable: $CP_BINARY" >&2; exit 1; }
}

cp_cg_step_print_summary() {
  echo "job: ${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-manual}"
  echo "nodes: ${SLURM_NODELIST:-manual}"
  echo "node: $(hostname)"
  echo "project_root: $CP_PROJECT_ROOT"
  echo "binary: $CP_BINARY"
  echo "result name: $CP_RESULT_NAME"
  echo "grid side: $CP_N"
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
  echo "I_MPI_FABRICS: ${I_MPI_FABRICS:-unset}"
  echo "FI_PROVIDER: ${FI_PROVIDER:-unset}"
  echo "NCCL_DEBUG: ${NCCL_DEBUG:-unset}"
  nvidia-smi || true
}

cp_cg_step_run_trials() {
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

cp_cg_step_main() {
  cp_cg_step_setup
  cp_cg_step_print_summary
  cp_cg_step_run_trials
}
