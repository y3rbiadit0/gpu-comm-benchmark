#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
binary="${repo_root}/build/cuda-mpi/implementations/cuda_mpi/cuda_mpi_vector_add"
nranks=${NP:-${SLURM_NTASKS:-4}}
n=${N:-1048576}

source "${repo_root}/cluster/leonardo/environment.sh" cuda
source "${repo_root}/cluster/leonardo/runtime/mpi-cuda.sh"

if [[ ! -x "$binary" ]]; then
  echo "no executable: $binary" >&2
  exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --cpu-freq=high -n "$nranks" "${repo_root}/cluster/leonardo/gpu-rank-wrapper.sh" "$binary" "$n"
else
  mpirun -np "$nranks" "${repo_root}/cluster/leonardo/gpu-rank-wrapper.sh" "$binary" "$n"
fi
