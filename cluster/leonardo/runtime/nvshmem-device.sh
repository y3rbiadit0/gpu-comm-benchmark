#!/usr/bin/env bash
set -euo pipefail

# Runtime for the cuda_nvshmem_device backend -- NVSHMEM's device API.
#
# Everything cuda_nvshmem runs with, except the one setting whose meaning
# changes when the reduction moves into the kernel.

# NCCL dispatch, which for this backend is not a choice.
#
# runtime/nvshmem.sh leaves NVSHMEM_DISABLE_NCCL=0, and for the host-initiated
# backend that is the right default: nvshmemx_double_sum_reduce_on_stream really
# does dispatch to NCCL there, the result is within 0.2% of cuda_nccl, and
# hiding that would hide a finding.
#
# There is no such dispatch from inside a kernel. nvshmem_double_sum_reduce and
# its _warp/_block forms are device functions; NCCL has no device entry point
# for them to call, so this backend runs NVSHMEM's own reduction path whatever
# this variable says. Setting it to 1 makes that explicit instead of leaving a
# 0 in the recorded environment that a later reader would take for a dispatched
# run.
#
# It also matches aCG, which sets NVSHMEM_DISABLE_NCCL=1 for every solver
# variant. That mismatch is exactly what made the host-NVSHMEM reduction
# unpredictable from the host-initiated benchmark -- the benchmark measured a
# NCCL-dispatched collective and the solver measured NVSHMEM's own. This
# backend is built to predict the device solver, so it must not repeat it.
#
# Set BEFORE sourcing nvshmem.sh, not after: that file assigns
# NVSHMEM_DISABLE_NCCL=${NVSHMEM_DISABLE_NCCL:-0}, so by the time it returns the
# variable is always set and a `:-1` here would never fire. Assigning first
# leaves its default inert and still lets the environment override both.
export NVSHMEM_DISABLE_NCCL=${NVSHMEM_DISABLE_NCCL:-1}

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/nvshmem.sh"
