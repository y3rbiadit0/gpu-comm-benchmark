# SYCL + MPI

SYCL examples using MPI for process launch, data movement, and result collection.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `sycl_mpi_vector_add` | Each rank computes `c[i] = a[i] + b[i]` for a contiguous global slice. | MPI distributes inputs and gathers local results. |
| `sycl_mpi_halo_1d` | Comm-only 1D halo exchange, periodic ring, swept halo width. | SYCL-aware `MPI_Isend`/`MPI_Irecv`/`MPI_Waitall` exchange USM device-buffer halos with both neighbors. |
| `sycl_mpi_dot_product` | Double-precision global dot product (CG inner-product). | `MPI_Allreduce` reduces a device-resident scalar (SYCL reduction) across ranks. |
| `sycl_mpi_pingpong` | Two-endpoint one-way latency/bandwidth, internal size sweep. | CUDA-aware `MPI_Send`/`MPI_Recv` round-trip USM device buffers between 2 ranks. |
| `sycl_mpi_halo_2d` | 2D 5-point Jacobi stencil, column-slab decomposition. | CUDA-aware `MPI_Sendrecv` exchanges packed (strided) halo columns with neighbors. |
| `sycl_mpi_alltoall` | All-to-all personalized exchange (bisection bandwidth). | CUDA-aware `MPI_Alltoall` over USM device buffers. |
| `sycl_mpi_cg_step` | CG iteration skeleton (SpMV halo + two reductions). | CUDA-aware `MPI_Sendrecv` halo + two `MPI_Allreduce` (SYCL reductions). |

## Run

```bash
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_vector_add 1048576
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_halo_1d 1048576 100 20
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_dot_product 1048576 100 20
mpirun -np 2 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_pingpong 4194304 100 20
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_halo_2d 4096 50 10
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_alltoall 65536 100 20
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_cg_step 512 50 10
```
