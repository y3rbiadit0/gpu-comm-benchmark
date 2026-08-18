#!/bin/bash -l
#SBATCH --job-name=halo_1d_sycl_oneccl_1n2g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

set -euo pipefail

GPU_BENCH_PROJECT_ROOT=${GPU_BENCH_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
GPU_BENCH_STACK=sycl
GPU_BENCH_RUNTIME=oneccl-nccl
GPU_BENCH_BINARY=${GPU_BENCH_BINARY:-$GPU_BENCH_PROJECT_ROOT/build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_halo_1d}
GPU_BENCH_RESULT_NAME=${GPU_BENCH_RESULT_NAME:-halo-1d-sycl-oneccl-1n2g}
GPU_BENCH_NODES=1
GPU_BENCH_TASKS_PER_NODE=2
GPU_BENCH_LAUNCHER=mpirun

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/halo_1d/common.sh"
gpu_bench_halo_1d_main
