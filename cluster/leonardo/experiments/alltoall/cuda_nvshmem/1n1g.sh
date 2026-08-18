#!/bin/bash -l
#SBATCH --job-name=alltoall_cuda_nvshmem_1n1g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail

GPU_BENCH_PROJECT_ROOT=${GPU_BENCH_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
GPU_BENCH_STACK=cuda
GPU_BENCH_RUNTIME=nvshmem
GPU_BENCH_BINARY=${GPU_BENCH_BINARY:-$GPU_BENCH_PROJECT_ROOT/build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_alltoall}
GPU_BENCH_RESULT_NAME=${GPU_BENCH_RESULT_NAME:-alltoall-cuda-nvshmem-1n1g}
GPU_BENCH_NODES=1
GPU_BENCH_TASKS_PER_NODE=1

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/alltoall/common.sh"
gpu_bench_alltoall_main
