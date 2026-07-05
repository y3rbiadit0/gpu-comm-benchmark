#!/usr/bin/env bash
set -euo pipefail

# Shared engine for all Leonardo experiments. Each experiment's common.sh is a
# thin shim that sources this file and provides:
#
#   CP_EXPERIMENT              results subdirectory name (e.g. dot_product)
#   CP_N_LABEL                 summary label for CP_N (e.g. "problem size")
#   cp_experiment_defaults     required; sets CP_N/CP_ITERS/CP_EXTRA_ARGS defaults
#   cp_experiment_extra_summary  optional; extra summary lines
#   cp_experiment_launch_prefix  optional override; echoes a per-rank command
#                                prefix spliced in front of the binary (e.g. nsys)
#   cp_<name>_main             wrapper around cp_experiment_main, called by the
#                              sbatch job scripts

cp_experiment_launch_prefix() { :; }

cp_experiment_setup() {
  : "${CP_EXPERIMENT:?missing CP_EXPERIMENT}"
  : "${CP_N_LABEL:?missing CP_N_LABEL}"
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

  CP_NTRIALS=${CP_NTRIALS:-3}
  CP_LAUNCHER=${CP_LAUNCHER:-srun}
  CP_RUN_DIR=${CP_RUN_DIR:-results/$CP_RESULT_NAME/$CP_EXPERIMENT}

  cp_experiment_defaults

  [ -x "$CP_BINARY" ] || { echo "no executable: $CP_BINARY" >&2; exit 1; }
}

cp_experiment_print_summary() {
  echo "job: ${SLURM_JOB_NAME:-manual}/${SLURM_JOB_ID:-manual}"
  echo "nodes: ${SLURM_NODELIST:-manual}"
  echo "node: $(hostname)"
  echo "project_root: $CP_PROJECT_ROOT"
  echo "binary: $CP_BINARY"
  echo "result name: $CP_RESULT_NAME"
  echo "$CP_N_LABEL: $CP_N"
  [[ -n "${CP_ITERS:-}" ]] && echo "iterations: $CP_ITERS"
  [[ -n "${CP_WARMUP:-}" ]] && echo "warmup: $CP_WARMUP"
  [[ -n "${CP_EXTRA_ARGS:-}" ]] && echo "extra args: $CP_EXTRA_ARGS"
  echo "trials: $CP_NTRIALS"
  echo "nodes requested: $CP_NODES"
  echo "tasks per node: $CP_TASKS_PER_NODE"
  echo "launcher: $CP_LAUNCHER"
  echo "launcher path: $(command -v "$CP_LAUNCHER" 2>/dev/null || true)"
  echo "stack: $CP_STACK"
  echo "runtime: $CP_RUNTIME"
  if declare -F cp_experiment_extra_summary >/dev/null; then
    cp_experiment_extra_summary
  fi
  cp_leonardo_print_env
  nvidia-smi || true
}

cp_experiment_run_trials() {
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
        $(cp_experiment_launch_prefix) \
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
        $(cp_experiment_launch_prefix) \
        "$CP_BINARY" \
        "$CP_N" \
        $CP_EXTRA_ARGS \
        >"${outfile}.tmp" 2>"${errfile}.tmp" \
      && mv --verbose "${outfile}.tmp" "$outfile" \
      && mv --verbose "${errfile}.tmp" "$errfile"
    fi

    # Back-to-back job steps can stall on Leonardo while the previous step
    # finalizes; give it a moment before launching the next srun.
    if [[ "$trial" -lt "$CP_NTRIALS" ]]; then
      sleep 1
    fi
  done
}

cp_experiment_main() {
  cp_experiment_setup
  cp_experiment_print_summary
  cp_experiment_run_trials
}
