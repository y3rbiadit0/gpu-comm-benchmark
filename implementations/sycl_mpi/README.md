# SYCL + MPI

SYCL examples using MPI for process launch, data movement, and result collection.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `sycl_mpi_vector_add` | Each rank computes `c[i] = a[i] + b[i]` for a contiguous global slice. | MPI distributes inputs and gathers local results. |
| `sycl_mpi_halo_1d` | One-step 1D stencil over contiguous rank-owned segments. | Host-buffer `MPI_Sendrecv` exchanges one boundary value with each neighbor. |

## Run

```bash
mpirun -np 4 ./build/sycl-mpi/implementations/sycl_mpi/sycl_mpi_vector_add 1048576
mpirun -np 4 ./build/sycl-mpi/implementations/sycl_mpi/sycl_mpi_halo_1d 1048576
```
