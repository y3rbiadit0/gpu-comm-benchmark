#!/bin/bash -l
#SBATCH --job-name=pingpong_cuda_mpi_1n2g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

set -euo pipefail

GPU_BENCH_PROJECT_ROOT=${GPU_BENCH_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
GPU_BENCH_STACK=cuda
GPU_BENCH_RUNTIME=mpi-cuda
GPU_BENCH_BINARY=${GPU_BENCH_BINARY:-$GPU_BENCH_PROJECT_ROOT/build/leonardo-cuda-mpi/src/mpi/cuda/cuda_mpi_pingpong}
GPU_BENCH_RESULT_NAME=${GPU_BENCH_RESULT_NAME:-pingpong-cuda-mpi-1n2g}
GPU_BENCH_NODES=1
GPU_BENCH_TASKS_PER_NODE=2

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/pingpong/common.sh"
gpu_bench_pingpong_main
