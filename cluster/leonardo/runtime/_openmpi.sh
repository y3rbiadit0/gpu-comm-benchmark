#!/usr/bin/env bash
# Shared baseline for the runtimes that go through Open MPI.
#
# Sourced by mpi-cuda.sh, sycl-mpi.sh, nvshmem.sh and oshmpi.sh -- every backend
# whose data plane is Open MPI + UCX. The two oneCCL runtimes do NOT source this:
# oneCCL has its own transport and none of these apply.
#
# This exists because the block below was copied into four files, and the four
# copies drifted. `OMPI_MCA_coll_ucc_enable` in particular was changed in some
# and not others, which is how a broken collective baseline survived a whole
# measurement round. One definition, four users.
#
# Leading underscore marks a non-runtime file, matching deps/_lib.sh; the backend
# registry never resolves a runtime name starting with one.

# Host-side placement. Slurm's default binding pins every rank of a task to one
# core, which starves the progress threads; `none` lets them spread.
export LC_ALL=${LC_ALL:-C}
export OMP_DISPLAY_ENV=${OMP_DISPLAY_ENV:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=${SLURM_CPU_BIND:-none}

# Open MPI data plane. btl ^openib because UCX owns InfiniBand here; hcoll off
# because it and UCC both claim the collective path and hcoll loses on A100.
export OMPI_MCA_btl=${OMPI_MCA_btl:-^openib}
export OMPI_MCA_coll_hcoll_enable=${OMPI_MCA_coll_hcoll_enable:-0}
export OMPI_MCA_opal_cuda_support=${OMPI_MCA_opal_cuda_support:-1}

# UCC is ON by default: it is transformative for large allreduce (139x at 16 MiB)
# and costs latency below ~64 KiB. Benchmarks in the small-message regime turn it
# off in their own common.sh, which wins because of the ${:-} default here.
# See cluster/leonardo/runtime/mpi-cuda.sh for the measured table.
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-1}
