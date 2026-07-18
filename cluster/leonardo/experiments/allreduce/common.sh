#!/usr/bin/env bash
set -euo pipefail

CP_EXPERIMENT=allreduce
CP_N_LABEL="max message elements"

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

# allreduce sweeps powers of two from 1 element up to CP_N by default;
# CP_MSG_SIZES can override that with a comma-separated list.
cp_experiment_defaults() {
  CP_N=${CP_N:-4194304}
  CP_ITERS=${CP_ITERS:-100}
  CP_WARMUP=${CP_WARMUP:-20}
  # allreduce binaries accept: <max_elements> [iterations] [warmup] [message_sizes]
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-"$CP_ITERS $CP_WARMUP"}
  if [[ -n "${CP_MSG_SIZES:-}" ]]; then
    CP_EXTRA_ARGS="$CP_EXTRA_ARGS $CP_MSG_SIZES"
  fi
}

cp_experiment_extra_summary() {
  echo "message sizes: ${CP_MSG_SIZES:-powers-of-two}"
}

cp_allreduce_main() { cp_experiment_main; }
