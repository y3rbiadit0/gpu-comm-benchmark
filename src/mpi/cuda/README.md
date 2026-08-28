# CUDA + MPI

This backend uses CUDA-aware MPI for GPU-resident communication and MPI for
process launch and result collection.

## Implemented operations

| Benchmark | MPI operation |
| --- | --- |
| `pingpong` | Blocking `MPI_Send`/`MPI_Recv` round trip |
| `halo_1d` | Persistent nonblocking neighbor sends and receives |
| `allreduce` | `MPI_Allreduce` |
| `alltoall` | `MPI_Alltoall` |
| `cg_step` | `MPI_Sendrecv` halo and two `MPI_Allreduce` calls |
| `moe` | Variable-count `MPI_Alltoallv` dispatch and combine |

## Build

Use the portable preset with a CUDA-aware MPI environment:

```bash
cmake --preset cuda-mpi
cmake --build --preset cuda-mpi
```

Leonardo uses the `leonardo-cuda-mpi` preset and the `cuda_mpi` harness backend.

See the [Leonardo guide](../../../cluster/leonardo/README.md) for its HPC-X
toolchain.

For benchmark semantics, see the
[benchmark contracts](../../../docs/README.md#benchmark-contracts). For
arguments, defaults, topologies, and launch examples, see the
[experiment operations](../../../cluster/harness/README.md#experiment-operations).
