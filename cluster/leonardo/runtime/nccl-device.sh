#!/usr/bin/env bash
set -euo pipefail

# Runtime for the cuda_nccl_device backend -- NCCL's device API.
#
# Everything cuda_nccl runs with, plus the settings that belong to the device
# API's transports. Only GIN has one that matters today, and it decides whether
# a GIN kernel completes at all on this machine; an LSA benchmark added beside
# it needs nothing here, because LSA has no doorbell to choose.

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/mpi-cuda.sh"

# Which side rings the NIC doorbell.
#
#   2  proxy  a CPU thread posts the descriptor the kernel wrote
#   3  GDAKI  the GPU posts it itself
#
# NCCL selects GDAKI on its own when the hardware reports GPUDirect RDMA, and
# Leonardo does report it: the GDAKI backend initializes completely -- queue
# pairs created and connected across nodes, memory regions registered, "NET/
# GIN_IB_GDAKI : GPU Direct RDMA Enabled" for all four HCAs -- and then the
# kernel never finishes. Everything the host sets up succeeds and only the
# device-initiated step goes nowhere, which is the GPU being unable to ring the
# doorbell. Measured 2026-09-08 with NCCL's own GIN alltoall example: proxy
# passes at 1n4g and 2n4g, GDAKI stalls at both.
#
# This is the same wall the machine puts in front of NVSHMEM's classic IBGDA,
# and the reason runtime/nvshmem.sh runs NVSHMEM_IBGDA_NIC_HANDLER=cpu. The two
# backends therefore measure the same compromise from two implementations.
#
# Left as an override rather than a hard assignment so GDAKI can be re-tested
# after a driver or kernel change -- but the default must be the proxy, because
# the failure mode is a hang that burns the whole allocation, not an error.
export NCCL_GIN_TYPE=${NCCL_GIN_TYPE:-2}
