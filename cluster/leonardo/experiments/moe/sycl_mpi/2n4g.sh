#!/bin/bash -l
#SBATCH --job-name=moe_sycl_mpi_2n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

set -euo pipefail

CP_PROJECT_ROOT=${CP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CP_STACK=sycl
CP_RUNTIME=sycl-mpi
CP_BINARY=${CP_BINARY:-$CP_PROJECT_ROOT/build/leonardo-sycl-mpi/src/mpi/sycl/sycl_mpi_moe}
CP_RESULT_NAME=${CP_RESULT_NAME:-moe-sycl-mpi-2n4g}
CP_NODES=2
CP_TASKS_PER_NODE=4

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/moe/common.sh"
cp_moe_main
