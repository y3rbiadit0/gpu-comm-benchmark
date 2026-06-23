# CUDA + MPI

CUDA examples using MPI for process launch, data movement, and result collection.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `cuda_mpi_vector_add` | Each rank computes `c[i] = a[i] + b[i]` for a contiguous global slice. | MPI distributes inputs and gathers local results. |
| `cuda_mpi_halo_1d` | One-step 1D stencil over contiguous rank-owned segments. | Host-buffer `MPI_Sendrecv` exchanges one boundary value with each neighbor. |
| `cuda_mpi_halo_1d_cuda_aware_iter` | Multi-iteration 1D halo stencil. | CUDA-aware `MPI_Sendrecv` exchanges device-buffer ghost cells each iteration. |
| `cuda_mpi_halo_1d_cuda_aware_persistent_iter` | Multi-iteration 1D halo stencil. | Persistent CUDA-aware MPI requests exchange device-buffer ghost cells. |

## Run

```bash
mpirun -np 4 ./build/cuda-mpi/src/cuda_mpi/cuda_mpi_vector_add 1048576
mpirun -np 4 ./build/cuda-mpi/src/cuda_mpi/cuda_mpi_halo_1d 1048576
mpirun -np 4 ./build/cuda-mpi/src/cuda_mpi/cuda_mpi_halo_1d_cuda_aware_iter 1048576 100
```
