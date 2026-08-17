#!/bin/bash -l
#SBATCH --job-name=pingpong_cuda_nvshmem_1n2g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

set -euo pipefail

CP_PROJECT_ROOT=${CP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CP_STACK=cuda
CP_RUNTIME=nvshmem
CP_BINARY=${CP_BINARY:-$CP_PROJECT_ROOT/build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_pingpong}
CP_RESULT_NAME=${CP_RESULT_NAME:-pingpong-cuda-nvshmem-1n2g}
CP_NODES=1
CP_TASKS_PER_NODE=2

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/pingpong/common.sh"
cp_pingpong_main
