#!/usr/bin/env bash
set -euo pipefail

local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${SLURM_LOCALID:-}}}
if [[ -n "$local_rank" ]]; then
  export CUDA_VISIBLE_DEVICES=$local_rank
fi

exec "$@"
