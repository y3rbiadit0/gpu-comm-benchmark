#!/bin/bash -l
#SBATCH --job-name=allreduce_sycl_oneccl_oshmpi_2n1g
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
CP_RUNTIME=oneccl-oshmpi
CP_BINARY=${CP_BINARY:-$CP_PROJECT_ROOT/build/leonardo-sycl-oneccl-oshmpi/src/xccl/sycl/sycl_oneccl_allreduce}
CP_RESULT_NAME=${CP_RESULT_NAME:-allreduce-sycl-oneccl-oshmpi-2n1g}
CP_NODES=2
CP_TASKS_PER_NODE=1
CP_LAUNCHER=mpirun

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/allreduce/common.sh"
cp_allreduce_main
