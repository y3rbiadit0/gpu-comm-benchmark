#!/usr/bin/env bash
set -euo pipefail

CP_EXPERIMENT=alltoall
CP_N_LABEL="count per peer"

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

# alltoall sends CP_N elements to every peer (send/recv buffers are ranks*CP_N).
cp_experiment_defaults() {
  CP_N=${CP_N:-65536}
  CP_ITERS=${CP_ITERS:-100}
  CP_WARMUP=${CP_WARMUP:-20}
  # alltoall binaries accept: <count_per_peer> [iterations] [warmup]
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-"$CP_ITERS $CP_WARMUP"}
}

cp_alltoall_main() { cp_experiment_main; }
