#!/usr/bin/env bash
set -euo pipefail

CP_EXPERIMENT=moe
CP_N_LABEL="tokens per rank"

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

cp_experiment_defaults() {
  CP_N=${CP_N:-16384}
  CP_HIDDEN=${CP_HIDDEN:-256}
  CP_ITERS=${CP_ITERS:-100}
  CP_WARMUP=${CP_WARMUP:-20}
  # MoE binaries accept: <tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]
  CP_EXTRA_ARGS="$CP_HIDDEN $CP_ITERS $CP_WARMUP"
  if [[ -n "${CP_ROUTINGS:-}" ]]; then
    CP_EXTRA_ARGS="$CP_EXTRA_ARGS $CP_ROUTINGS"
  fi
}

cp_experiment_extra_summary() {
  echo "hidden size: $CP_HIDDEN"
  echo "routing cases: ${CP_ROUTINGS:-uniform,locality80,hotspot80}"
}

cp_moe_main() { cp_experiment_main; }
