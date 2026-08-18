#!/bin/bash -l
#SBATCH --job-name=moe_sycl_oneccl_1n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

set -euo pipefail

GPU_BENCH_PROJECT_ROOT=${GPU_BENCH_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
GPU_BENCH_STACK=sycl
GPU_BENCH_RUNTIME=oneccl-nccl
GPU_BENCH_BINARY=${GPU_BENCH_BINARY:-$GPU_BENCH_PROJECT_ROOT/build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_moe}
GPU_BENCH_RESULT_NAME=${GPU_BENCH_RESULT_NAME:-moe-sycl-oneccl-1n4g}
GPU_BENCH_NODES=1
GPU_BENCH_TASKS_PER_NODE=4
GPU_BENCH_LAUNCHER=mpirun

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/moe/common.sh"
gpu_bench_moe_main
