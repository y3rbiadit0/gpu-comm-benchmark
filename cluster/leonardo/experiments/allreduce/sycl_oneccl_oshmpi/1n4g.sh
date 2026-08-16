#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=allreduce_sycl_oneccl_oshmpi_1n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

set -euo pipefail

CP_PROJECT_ROOT=${CP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CP_STACK=sycl
CP_RUNTIME=oneccl-oshmpi
CP_BINARY=${CP_BINARY:-$CP_PROJECT_ROOT/build/leonardo-sycl-oneccl-oshmpi/src/xccl/sycl/sycl_oneccl_allreduce}
CP_RESULT_NAME=${CP_RESULT_NAME:-allreduce-sycl-oneccl-oshmpi-1n4g}
CP_NODES=1
CP_TASKS_PER_NODE=4
CP_LAUNCHER=mpirun

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/allreduce/common.sh"
cp_allreduce_main
