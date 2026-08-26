#!/usr/bin/env bash

# SLURM submission defaults for Leonardo, in one place rather than repeated in
# every job script.
#
# sbatch reads SBATCH_ACCOUNT and SBATCH_PARTITION from the environment, so the
# job scripts carry no -A/-p directives and anyone can point them at their own
# allocation:
#
#   GPU_BENCH_SLURM_ACCOUNT=IscrC_OTHER cluster/leonardo/launch.sh --all allreduce
#
# Precedence is the usual SLURM one - an explicit `sbatch --account=...` still
# wins over anything set here.
#
# Sourced by cluster/leonardo/launch.sh --all and by cluster/leonardo/environment.sh, so both
# `make submit` and a hand-run sbatch from a prepared shell pick it up.

export SBATCH_ACCOUNT=${GPU_BENCH_SLURM_ACCOUNT:-${SBATCH_ACCOUNT:-IscrC_HIGRAPH_0}}
export SBATCH_PARTITION=${GPU_BENCH_SLURM_PARTITION:-${SBATCH_PARTITION:-boost_usr_prod}}
