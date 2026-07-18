# CUDA + MPI

CUDA examples using MPI for process launch, data movement, and result collection.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `cuda_mpi_halo_1d` | Comm-only 1D halo exchange, periodic ring, swept halo width. | CUDA-aware `MPI_Isend`/`MPI_Irecv`/`MPI_Waitall` exchange device-buffer halos with both neighbors. |
| `cuda_mpi_pingpong` | Two-endpoint one-way latency/bandwidth, internal size sweep. | CUDA-aware `MPI_Send`/`MPI_Recv` round-trip device buffers between 2 ranks. |
| `cuda_mpi_allreduce` | Float32 sum allreduce latency/bandwidth, internal size sweep. | CUDA-aware `MPI_Allreduce` over device buffers. |
| `cuda_mpi_alltoall` | All-to-all personalized exchange (bisection bandwidth). | CUDA-aware `MPI_Alltoall` over device buffers. |
| `cuda_mpi_cg_step` | CG iteration skeleton (SpMV halo + two reductions). | CUDA-aware `MPI_Sendrecv` halo + two `MPI_Allreduce`. |
| `cuda_mpi_moe` | Top-1 MoE dispatch + combine with variable expert loads. | Two CUDA-aware `MPI_Alltoallv` operations over device buffers: variable-count dispatch followed by inverse combine. |

## Run

```bash
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_halo_1d 1048576 100 20
mpirun -np 2 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_pingpong 4194304 100 20
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_allreduce 4194304 100 20
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_alltoall 65536 100 20
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_cg_step 512 50 10
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_moe 16384 256 100 20 uniform,locality80,hotspot80
```
