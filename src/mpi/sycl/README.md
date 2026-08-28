# SYCL + MPI

This backend combines SYCL USM device buffers with GPU-aware MPI. MPI provides
process launch, data movement, reductions, and result collection.

| Benchmark | MPI operation |
| --- | --- |
| `pingpong` | Blocking `MPI_Send`/`MPI_Recv` round trip |
| `halo_1d` | Persistent nonblocking neighbor sends and receives |
| `allreduce` | `MPI_Allreduce` |
| `alltoall` | `MPI_Alltoall` |
| `cg_step` | `MPI_Sendrecv` halo and two `MPI_Allreduce` calls |
| `moe` | Variable-count `MPI_Alltoallv` dispatch and combine |

## Build And Run

Use the portable preset after loading a SYCL compiler and compatible GPU-aware
MPI:

```bash
cmake --preset sycl-mpi
cmake --build --preset sycl-mpi
```

Leonardo uses `leonardo-sycl-mpi` and the `sycl_mpi` harness backend:

```bash
cluster/harness/launch.sh halo_1d sycl_mpi 1n4g
```

See the [Leonardo guide](../../../cluster/leonardo/README.md) for its DPC++ and
HPC-X setup.
