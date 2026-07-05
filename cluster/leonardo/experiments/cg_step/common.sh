#!/usr/bin/env bash
set -euo pipefail

CP_EXPERIMENT=cg_step
CP_N_LABEL="grid side"

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

# cg_step works on a square CP_N x CP_N grid; CP_N is the side length.
# Kept small so the halo exchange + two reductions dominate over stencil compute.
cp_experiment_defaults() {
  CP_N=${CP_N:-512}
  CP_ITERS=${CP_ITERS:-50}
  CP_WARMUP=${CP_WARMUP:-10}
  # cg_step binaries accept: <side> [iterations] [warmup]
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-"$CP_ITERS $CP_WARMUP"}
}

cp_cg_step_main() { cp_experiment_main; }
