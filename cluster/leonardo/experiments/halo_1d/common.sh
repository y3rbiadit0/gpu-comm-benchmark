#!/usr/bin/env bash
set -euo pipefail

CP_EXPERIMENT=halo_1d
CP_N_LABEL="problem size"

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

cp_experiment_defaults() {
  CP_N=${CP_N:-1048576}
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-}
  # Opt-in per-rank profiling. CP_PROFILE=nsys wraps each rank in Nsight Systems
  # and writes one .nsys-rep per rank under $CP_RUN_DIR/profiles. Profiling
  # perturbs timing, so use a dedicated CP_NTRIALS=1 run for the timeline and
  # never report numbers from a profiled run. See docs/analysis/halo_1d-crossover.md.
  CP_PROFILE=${CP_PROFILE:-}
  CP_NSYS_TRACE=${CP_NSYS_TRACE:-cuda,nvtx,mpi}
}

cp_experiment_extra_summary() {
  echo "profile: ${CP_PROFILE:-off}"
}

# Echoes the per-rank profiler command prefix to splice in front of the binary,
# or nothing when profiling is off. Each rank gets its own .nsys-rep via the
# launcher's per-rank env (SLURM_PROCID under srun, OMPI_COMM_WORLD_RANK under
# mpirun); nsys substitutes %q{VAR} at runtime, so the token must stay unquoted.
cp_experiment_launch_prefix() {
  [[ "${CP_PROFILE:-}" == "nsys" ]] || return 0

  local rank_token='%q{SLURM_PROCID}'
  [[ "$CP_LAUNCHER" == "mpirun" ]] && rank_token='%q{OMPI_COMM_WORLD_RANK}'

  mkdir -p "$CP_RUN_DIR/profiles"
  local out="$CP_RUN_DIR/profiles/${SLURM_JOB_NAME:-manual}-${SLURM_JOB_ID:-manual}-${trial}-rank${rank_token}"

  printf '%s ' nsys profile --force-overwrite=true --trace="$CP_NSYS_TRACE" \
    --sample=none --cpuctxsw=none --output="$out"
}

cp_halo_1d_main() { cp_experiment_main; }
