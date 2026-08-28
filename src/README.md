# Backend Implementations

This directory contains equivalent implementations of every benchmark across
the supported communication models. Benchmark semantics are defined by the
[benchmark contracts](../docs/README.md#benchmark-contracts); these pages
document how each backend expresses and completes those operations.

| Model | Programming model | Communication library | Implementation |
| --- | --- | --- | --- |
| MPI | CUDA | CUDA-aware MPI | [`mpi/cuda`](mpi/cuda/README.md) |
| MPI | SYCL | GPU-aware MPI | [`mpi/sycl`](mpi/sycl/README.md) |
| XCCL | CUDA | NCCL | [`xccl/cuda`](xccl/cuda/README.md) |
| XCCL | SYCL | oneCCL | [`xccl/sycl`](xccl/sycl/README.md) |
| SHMEM | CUDA | NVSHMEM | [`shmem/nvshmem`](shmem/nvshmem/README.md) |
| SHMEM | CUDA | OSHMPI | [`shmem/oshmpi`](shmem/oshmpi/README.md) |

Each backend has the same source tiers:

| Directory | Contents |
| --- | --- |
| `microbench/` | `pingpong`, `halo_1d`, `allreduce`, and `alltoall` |
| `application/` | `cg_step` and `moe` application patterns |

Runtime backend names and declared benchmark coverage are listed in the
[support matrix](../docs/reference/support-matrix.md). For defaults, topology
constraints, and launch examples, use the
[experiment operations](../cluster/harness/README.md#experiment-operations).
