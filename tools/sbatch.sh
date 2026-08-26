#!/usr/bin/env bash
# Submit one gpu-comm-bench job with the project's SLURM defaults applied.
#
# The job scripts deliberately carry no -A/-p directives so anyone can point
# them at their own allocation (see cluster/leonardo/slurm.sh). Those defaults
# live in environment variables, which sbatch reads from *its own* environment -
# so `tools/launch.sh` gets them (it sources slurm.sh in-process) while a
# hand-run `sbatch cluster/.../1n4g.sh` from a fresh login shell silently does
# not, and lands on the cluster's default partition instead.
#
# This wrapper closes that gap for single-job runs:
#
#   tools/launch.sh allreduce oshmpi 1n4g
#   GPU_BENCH_N=17 tools/launch.sh halo_1d cuda_mpi 2n4g
#
# Everything is passed straight through, so sbatch flags still work and still
# win over the defaults (command line > environment > script directives):
#
#   tools/sbatch.sh --qos=boost_qos_dbg cluster/.../2n4g.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cluster/leonardo/slurm.sh"

if [[ $# -eq 0 ]]; then
  echo "usage: tools/sbatch.sh [sbatch options] <job script>" >&2
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "error: sbatch not found -- run this on Leonardo (a login node)." >&2
  exit 1
fi

echo "account: $SBATCH_ACCOUNT   partition: $SBATCH_PARTITION" >&2
exec sbatch "$@"
