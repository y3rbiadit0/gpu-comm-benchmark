#!/usr/bin/env bash

# Where everything this project builds lives. One definition, read by the
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

GPU_BENCH_WORK_ROOT=${GPU_BENCH_WORK_ROOT:-${SCRATCH:?set SCRATCH or GPU_BENCH_WORK_ROOT to a build filesystem}/gpu-comm-bench}
GPU_BENCH_PREFIX_ROOT=${GPU_BENCH_PREFIX_ROOT:-$HOME/opt/gpu-comm-bench}

# Sources and build trees - scratch.
export GPU_BENCH_SRC_DIR=${GPU_BENCH_SRC_DIR:-$GPU_BENCH_WORK_ROOT/src}
export GPU_BENCH_BUILD_DIR=${GPU_BENCH_BUILD_DIR:-$GPU_BENCH_WORK_ROOT/build}

# Install prefixes - persistent. One target under deps/ produces each of these, and
# runtime/*.sh resolves them by these names. Nothing else may define them: two
# definitions with different defaults resolve by source order, which is how a build
# silently links one install's headers against another's libraries.
#
# Vendor-supplied libraries are not listed here. NVSHMEM, for one, comes from the
# nvhpc module and belongs to env/cuda.sh.
export OSHMPI_HOME=${OSHMPI_HOME:-$GPU_BENCH_PREFIX_ROOT/oshmpi}
export ONECCL_OSHMPI_ROOT=${ONECCL_OSHMPI_ROOT:-$GPU_BENCH_PREFIX_ROOT/oneccl-oshmpi}
# No deps/ target builds this one yet - it was installed by hand - so the default is
# where it actually is rather than where the layout would put it. Moves to
# $GPU_BENCH_PREFIX_ROOT/oneccl-nccl when deps/oneccl-nccl.sh exists.
export ONECCL_NCCL_ROOT=${ONECCL_NCCL_ROOT:-$HOME/opt/oneccl-nccl-leonardo}

export GPU_BENCH_WORK_ROOT GPU_BENCH_PREFIX_ROOT
