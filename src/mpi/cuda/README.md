# CUDA + MPI

This backend uses CUDA-aware MPI for GPU-resident communication and MPI for
process launch and result collection.

| Benchmark | MPI operation |
| --- | --- |
| `pingpong` | Blocking `MPI_Send`/`MPI_Recv` round trip |
| `halo_1d` | Persistent nonblocking neighbor sends and receives |
| `allreduce` | `MPI_Allreduce` |
| `alltoall` | `MPI_Alltoall` |
| `cg_step` | `MPI_Sendrecv` halo and two `MPI_Allreduce` calls |
| `moe` | Variable-count `MPI_Alltoallv` dispatch and combine |

## Build And Run

Use the portable preset with a CUDA-aware MPI environment:

```bash
cmake --preset cuda-mpi
cmake --build --preset cuda-mpi
```

Leonardo uses `leonardo-cuda-mpi` and the `cuda_mpi` harness backend:

```bash
cluster/harness/launch.sh halo_1d cuda_mpi 1n4g
```

See the [Leonardo guide](../../../cluster/leonardo/README.md) for its HPC-X
toolchain and the [experiment guides](../../../cluster/harness/experiments) for
arguments and defaults.
