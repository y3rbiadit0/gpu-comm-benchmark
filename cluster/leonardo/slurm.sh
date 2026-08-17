#!/usr/bin/env bash

# SLURM submission defaults for Leonardo, in one place rather than repeated in
# every job script.
#
# sbatch reads SBATCH_ACCOUNT and SBATCH_PARTITION from the environment, so the
# job scripts carry no -A/-p directives and anyone can point them at their own
# allocation:
#
#   CP_SLURM_ACCOUNT=IscrC_OTHER tools/submit_all.sh allreduce
#
# Precedence is the usual SLURM one - an explicit `sbatch --account=...` still
# wins over anything set here.
#
# Sourced by tools/submit_all.sh and by cluster/leonardo/environment.sh, so both
# `make submit` and a hand-run sbatch from a prepared shell pick it up.

export SBATCH_ACCOUNT=${CP_SLURM_ACCOUNT:-${SBATCH_ACCOUNT:-IscrC_HIGRAPH_0}}
export SBATCH_PARTITION=${CP_SLURM_PARTITION:-${SBATCH_PARTITION:-boost_usr_prod}}
