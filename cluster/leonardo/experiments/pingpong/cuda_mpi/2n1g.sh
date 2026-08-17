#!/bin/bash -l
#SBATCH --job-name=pingpong_cuda_mpi_2n1g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail

CP_PROJECT_ROOT=${CP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CP_STACK=cuda
CP_RUNTIME=mpi-cuda
CP_BINARY=${CP_BINARY:-$CP_PROJECT_ROOT/build/leonardo-cuda-mpi/src/mpi/cuda/cuda_mpi_pingpong}
CP_RESULT_NAME=${CP_RESULT_NAME:-pingpong-cuda-mpi-2n1g}
CP_NODES=2
CP_TASKS_PER_NODE=1

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/pingpong/common.sh"
cp_pingpong_main
