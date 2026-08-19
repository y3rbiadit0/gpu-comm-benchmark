#!/bin/bash -l
#SBATCH --job-name=oneccl_oshmpi
#SBATCH --output=./%x-%j-stdout.txt
#SBATCH --error=./%x-%j-stderr.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

# Runs oneCCL's OSHMPI test suite on Leonardo. Topology overrides work as usual:
# sbatch --nodes=... --ntasks-per-node=... --gres=gpu:...
#
#   ONECCL_SRC=$HOME/opt-src/oneCCL-oshmpi \
#     sbatch cluster/leonardo/oneccl-oshmpi/validate.sbatch

set -euo pipefail

project_root=${GPU_BENCH_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
oneccl_build_dir=${ONECCL_BUILD_DIR:-$SCRATCH/oneccl-oshmpi}

source "$project_root/cluster/leonardo/environment.sh" sycl
export GPU_BENCH_JOB_NODES=${SLURM_NNODES:-1}
source "$project_root/cluster/leonardo/runtime/oshmpi.sh"

# oneCCL builds with CMAKE_SKIP_RPATH, so nothing records where libccl.so.1 or
# liboshmpi.so live. Put both on the loader path explicitly.
export LD_LIBRARY_PATH="$oneccl_build_dir/src:$OSHMPI_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CCL_BACKEND=oshmpi
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-info}

# The runtime script exports a production-sized arena. Override it deliberately:
# the point of this job is to force every collective through the chunking path,
# which a 64M arena would skip entirely at these payload sizes. 128 bytes is the
# smallest value that still yields two aligned lanes.
export CCL_OSHMPI_STAGING_SIZE=${ONECCL_VALIDATE_STAGING_SIZE:-128}

ranks=$(( ${SLURM_NNODES:-1} * ${SLURM_NTASKS_PER_NODE:-2} ))

run_test() {
    local label=$1 binary="$oneccl_build_dir/tests/oshmpi/$2"
    if [[ ! -x "$binary" ]]; then
        printf '%s test not built, skipping\n\n' "$label"
        return 0
    fi
    printf -- '---- %s ----\n' "$label"
    local rc=0
    srun --cpu-freq=high -n "$ranks" "$binary" || rc=$?
    printf 'RESULT %s exit=%d\n\n' "$label" "$rc"
    return "$rc"
}

host_rc=0
sycl_rc=0
run_test host oshmpi_collectives_test || host_rc=$?
run_test sycl oshmpi_sycl_collectives_test || sycl_rc=$?

exit $(( host_rc != 0 || sycl_rc != 0 ))
