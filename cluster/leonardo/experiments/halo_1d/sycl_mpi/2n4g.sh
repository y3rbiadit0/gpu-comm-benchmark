#!/bin/bash -l
#SBATCH --job-name=halo_1d_sycl_mpi_2n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

set -euo pipefail

GPU_BENCH_PROJECT_ROOT=${GPU_BENCH_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
GPU_BENCH_STACK=sycl
GPU_BENCH_RUNTIME=sycl-mpi
GPU_BENCH_BINARY=${GPU_BENCH_BINARY:-$GPU_BENCH_PROJECT_ROOT/build/leonardo-sycl-mpi/src/mpi/sycl/sycl_mpi_halo_1d}
GPU_BENCH_RESULT_NAME=${GPU_BENCH_RESULT_NAME:-halo-1d-sycl-mpi-2n4g}
GPU_BENCH_NODES=2
GPU_BENCH_TASKS_PER_NODE=4

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/halo_1d/common.sh"
gpu_bench_halo_1d_main
