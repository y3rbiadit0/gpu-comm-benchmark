#!/usr/bin/env bash
set -euo pipefail

CP_EXPERIMENT=halo_1d
CP_N_LABEL="problem size"

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

cp_experiment_defaults() {
  CP_N=${CP_N:-1048576}
  CP_ITERS=${CP_ITERS:-100}
  CP_WARMUP=${CP_WARMUP:-20}
  # halo_1d binaries accept: <max_halo_elems> [iterations] [warmup] [halo_sizes]
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-"$CP_ITERS $CP_WARMUP"}
  if [[ -n "${CP_MSG_SIZES:-}" ]]; then
    CP_EXTRA_ARGS="$CP_EXTRA_ARGS $CP_MSG_SIZES"
  fi
}

cp_experiment_extra_summary() {
  echo "message sizes: ${CP_MSG_SIZES:-powers-of-two}"
}

cp_halo_1d_main() { cp_experiment_main; }
