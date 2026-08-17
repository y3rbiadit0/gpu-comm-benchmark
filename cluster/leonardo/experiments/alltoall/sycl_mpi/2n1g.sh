#!/bin/bash -l
#SBATCH --job-name=alltoall_sycl_mpi_2n1g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail

CP_PROJECT_ROOT=${CP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CP_STACK=sycl
CP_RUNTIME=sycl-mpi
CP_BINARY=${CP_BINARY:-$CP_PROJECT_ROOT/build/leonardo-sycl-mpi/src/mpi/sycl/sycl_mpi_alltoall}
CP_RESULT_NAME=${CP_RESULT_NAME:-alltoall-sycl-mpi-2n1g}
CP_NODES=2
CP_TASKS_PER_NODE=1

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/alltoall/common.sh"
cp_alltoall_main
