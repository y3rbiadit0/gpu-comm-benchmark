#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=halo_1d_cuda_nvshmem_optimized_2n1g
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
CP_RUNTIME=nvshmem
CP_BINARY=${CP_BINARY:-$CP_PROJECT_ROOT/build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_halo_1d_optimized}
CP_RESULT_NAME=${CP_RESULT_NAME:-halo-1d-cuda-nvshmem-optimized-2n1g}
CP_NODES=2
CP_TASKS_PER_NODE=1

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/halo_1d/common.sh"
cp_halo_1d_main
