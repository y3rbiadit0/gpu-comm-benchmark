#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=alltoall_oshmpi_2n4g
#SBATCH --error=./logs/%x-%j-stderr.txt
#SBATCH --output=./logs/%x-%j-stdout.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

set -euo pipefail

CP_PROJECT_ROOT=${CP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CP_STACK=cuda
CP_RUNTIME=oshmpi
CP_BINARY=${CP_BINARY:-$CP_PROJECT_ROOT/build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_alltoall}
CP_RESULT_NAME=${CP_RESULT_NAME:-alltoall-oshmpi-2n4g}
CP_NODES=2
CP_TASKS_PER_NODE=4

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/alltoall/common.sh"
cp_alltoall_main
