#!/usr/bin/env bash

# Where everything the playground builds lives. One definition, read by the
# bootstrap targets that produce these paths and by the runtime scripts that
# consume them, so the two cannot drift and nothing has to be overridden by hand.
#
# Two roots, because they have different lifetimes:
#
#   GPU_BENCH_WORK_ROOT    clones and build trees. Large, disposable, on scratch.
#                   Safe to purge; bootstrap recreates it.
#   GPU_BENCH_PREFIX_ROOT  install prefixes. Small, persistent, on $HOME, because jobs
#                   resolve libraries from here at run time.
#
# Relocate everything by setting one of those. Individual paths can still be
# overridden for one-off experiments, and are respected if already set.

GPU_BENCH_WORK_ROOT=${GPU_BENCH_WORK_ROOT:-${SCRATCH:?set SCRATCH or GPU_BENCH_WORK_ROOT to a build filesystem}/comm-playground}
GPU_BENCH_PREFIX_ROOT=${GPU_BENCH_PREFIX_ROOT:-$HOME/opt/comm-playground}

# Sources and build trees - scratch.
export GPU_BENCH_SRC_DIR=${GPU_BENCH_SRC_DIR:-$GPU_BENCH_WORK_ROOT/src}
export GPU_BENCH_BUILD_DIR=${GPU_BENCH_BUILD_DIR:-$GPU_BENCH_WORK_ROOT/build}

# Install prefixes - persistent. These names are what runtime/*.sh resolve.
export OSHMPI_HOME=${OSHMPI_HOME:-$GPU_BENCH_PREFIX_ROOT/oshmpi}
export ONECCL_OSHMPI_ROOT=${ONECCL_OSHMPI_ROOT:-$GPU_BENCH_PREFIX_ROOT/oneccl-oshmpi}
export ONECCL_NCCL_ROOT=${ONECCL_NCCL_ROOT:-$GPU_BENCH_PREFIX_ROOT/oneccl-nccl}
export NVSHMEM_HOME=${NVSHMEM_HOME:-$GPU_BENCH_PREFIX_ROOT/nvshmem}

export GPU_BENCH_WORK_ROOT GPU_BENCH_PREFIX_ROOT
