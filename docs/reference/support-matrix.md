# Support Matrix

The suite contains six source backends. Leonardo additionally builds oneCCL
against two transports, producing two runtime backends from the same SYCL source.

| Harness backend | Source | Communication library | Leonardo preset |
| --- | --- | --- | --- |
| `cuda_mpi` | [`src/mpi/cuda`](../../src/mpi/cuda/README.md) | HPC-X MPI | `leonardo-cuda-mpi` |
| `sycl_mpi` | [`src/mpi/sycl`](../../src/mpi/sycl/README.md) | HPC-X MPI | `leonardo-sycl-mpi` |
| `cuda_nccl` | [`src/xccl/cuda`](../../src/xccl/cuda/README.md) | NCCL | `leonardo-cuda-nccl` |
| `cuda_nvshmem` | [`src/shmem/nvshmem`](../../src/shmem/nvshmem/README.md) | NVSHMEM | `leonardo-cuda-nvshmem` |
| `oshmpi` | [`src/shmem/oshmpi`](../../src/shmem/oshmpi/README.md) | OSHMPI | `leonardo-oshmpi` |
| `sycl_oneccl` | [`src/xccl/sycl`](../../src/xccl/sycl/README.md) | oneCCL with NCCL | `leonardo-sycl-oneccl` |
| `sycl_oneccl_oshmpi` | [`src/xccl/sycl`](../../src/xccl/sycl/README.md) | oneCCL with OSHMPI | `leonardo-sycl-oneccl-oshmpi` |

## Declared Experiment Coverage

The harness matrix is the executable source of truth. A check means the backend
is declared for that benchmark; it does not replace validation of a particular
library build or transport.

| Benchmark | MPI, NCCL, NVSHMEM, OSHMPI, oneCCL/NCCL | oneCCL/OSHMPI |
| --- | :---: | :---: |
| [`pingpong`](../benchmarks/pingpong.md) | Yes | No |
| [`halo_1d`](../benchmarks/halo-1d.md) | Yes | No |
| [`allreduce`](../benchmarks/allreduce.md) | Yes | Yes |
| [`alltoall`](../benchmarks/alltoall.md) | Yes | Yes |
| [`cg_step`](../benchmarks/cg-step.md) | Yes | Yes |
| [`moe`](../benchmarks/moe.md) | Yes | No |

The common column represents `cuda_mpi`, `sycl_mpi`, `cuda_nccl`,
`cuda_nvshmem`, `oshmpi`, and `sycl_oneccl`.

| Benchmark | Declared topologies |
| --- | --- |
| `pingpong` | `1n2g`, `2n1g` |
| `halo_1d` | `1n2g`, `1n4g`, `2n1g`, `2n4g`, `4n4g`, `8n4g` |
| `allreduce`, `alltoall`, `moe`, `cg_step` | `1n1g`, `1n2g`, `1n4g`, `2n1g`, `2n4g`, `4n4g`, `8n4g` |

`pingpong` requires exactly two endpoints. `halo_1d` requires at least two ranks
for its periodic ring. The collective and application benchmarks retain a
single-rank control.

## Capability Handling

Library capability depends on the installed version and transport. In
particular, oneCCL point-to-point and variable-count operations must be checked
on the target system before a large sweep. A recognized unsupported MoE
point-to-point operation emits `NOT_IMPLEMENTED`/`SKIP`; hangs and unexpected
backend failures are errors and are bounded by the harness trial timeout.

Some implementations intentionally express an operation through native
primitives rather than a dedicated collective:

| Backend | Operation | Implementation model |
| --- | --- | --- |
| NCCL | `alltoall` | Grouped point-to-point sends and receives |
| NCCL | `halo_1d` | Grouped neighbor sends and receives |
| OSHMPI | `alltoall` | Per-peer one-sided puts plus completion barrier |
| OSHMPI | `halo_1d` | Nonblocking puts, device completion, and barrier |
| NVSHMEM | Neighbor patterns | Device or host initiated operations as documented by each benchmark |

These are measured implementations, not silent fallbacks. Their synchronization
costs remain inside the timed operation and must be considered when comparing
results.
