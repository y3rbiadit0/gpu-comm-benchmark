#!/bin/bash -l
#SBATCH --job-name=moe_cuda_nccl_1n1g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail

CP_PROJECT_ROOT=${CP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CP_STACK=cuda
CP_RUNTIME=mpi-cuda
CP_BINARY=${CP_BINARY:-$CP_PROJECT_ROOT/build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_moe}
CP_RESULT_NAME=${CP_RESULT_NAME:-moe-cuda-nccl-1n1g}
CP_NODES=1
CP_TASKS_PER_NODE=1

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/moe/common.sh"
cp_moe_main
