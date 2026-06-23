#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=vector_add_cuda_nccl_1n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

set -euo pipefail

CP_PROJECT_ROOT=${CP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CP_STACK=cuda
CP_RUNTIME=mpi-cuda
CP_BINARY=${CP_BINARY:-$CP_PROJECT_ROOT/build/leonardo-cuda-nccl/src/cuda_nccl/cuda_nccl_vector_add}
CP_RESULT_NAME=${CP_RESULT_NAME:-vector-add-cuda-nccl-1n4g}
CP_NODES=1
CP_TASKS_PER_NODE=4

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/vector_add/common.sh"
cp_vector_add_main
