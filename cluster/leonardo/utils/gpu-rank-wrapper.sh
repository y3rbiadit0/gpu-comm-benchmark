#!/usr/bin/env bash
set -euo pipefail

local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${SLURM_LOCALID:-}}}
if [[ -n "$local_rank" ]]; then
  export CUDA_VISIBLE_DEVICES=$local_rank
fi

global_rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${SLURM_PROCID:-}}}}
if [[ -n "$global_rank" ]]; then
  export GPU_BENCH_GLOBAL_RANK=$global_rank
elif [[ -n ${GPU_BENCH_PROFILE:-} ]]; then
  echo "unable to determine the global rank for profiler output" >&2
  exit 1
fi

exec "$@"
