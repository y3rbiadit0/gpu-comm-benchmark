#!/bin/bash -l
#SBATCH --job-name=moe_cuda_mpi_2n1g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail

GPU_BENCH_PROJECT_ROOT=${GPU_BENCH_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
GPU_BENCH_STACK=cuda
GPU_BENCH_RUNTIME=mpi-cuda
GPU_BENCH_BINARY=${GPU_BENCH_BINARY:-$GPU_BENCH_PROJECT_ROOT/build/leonardo-cuda-mpi/src/mpi/cuda/cuda_mpi_moe}
GPU_BENCH_RESULT_NAME=${GPU_BENCH_RESULT_NAME:-moe-cuda-mpi-2n1g}
GPU_BENCH_NODES=2
GPU_BENCH_TASKS_PER_NODE=1

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/moe/common.sh"
gpu_bench_moe_main
