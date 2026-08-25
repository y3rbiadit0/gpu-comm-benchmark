#!/bin/bash -l
#SBATCH --job-name=moe_cuda_nvshmem_4n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:15:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

set -euo pipefail

GPU_BENCH_PROJECT_ROOT=${GPU_BENCH_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
GPU_BENCH_STACK=cuda
GPU_BENCH_RUNTIME=nvshmem
GPU_BENCH_BINARY=${GPU_BENCH_BINARY:-$GPU_BENCH_PROJECT_ROOT/build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_moe}
GPU_BENCH_RESULT_NAME=${GPU_BENCH_RESULT_NAME:-moe-cuda-nvshmem-4n4g}
GPU_BENCH_NODES=4
GPU_BENCH_TASKS_PER_NODE=4

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/moe/common.sh"
gpu_bench_moe_main
