#!/bin/bash -l
#SBATCH --job-name=cg_step_sycl_oneccl_oshmpi_8n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:20:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

set -euo pipefail

GPU_BENCH_PROJECT_ROOT=${GPU_BENCH_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
GPU_BENCH_STACK=sycl
GPU_BENCH_RUNTIME=oneccl-oshmpi
GPU_BENCH_BINARY=${GPU_BENCH_BINARY:-$GPU_BENCH_PROJECT_ROOT/build/leonardo-sycl-oneccl-oshmpi/src/xccl/sycl/sycl_oneccl_cg_step}
GPU_BENCH_RESULT_NAME=${GPU_BENCH_RESULT_NAME:-cg-step-sycl-oneccl-oshmpi-8n4g}
GPU_BENCH_NODES=8
GPU_BENCH_TASKS_PER_NODE=4
GPU_BENCH_LAUNCHER=mpirun

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/cg_step/common.sh"
gpu_bench_cg_step_main
