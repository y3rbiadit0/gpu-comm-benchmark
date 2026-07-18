# CUDA + NCCL

CUDA examples using MPI for process launch and NCCL for GPU-resident communication.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `cuda_nccl_halo_1d` | Comm-only 1D halo exchange, periodic ring, swept halo width. | Grouped `ncclSend`/`ncclRecv` exchange device-buffer halos with both neighbors. |
| `cuda_nccl_pingpong` | Two-endpoint one-way latency/bandwidth, internal size sweep. | Matched `ncclSend`/`ncclRecv` round-trip device buffers between 2 ranks. |
| `cuda_nccl_allreduce` | Float32 sum allreduce latency/bandwidth, internal size sweep. | `ncclAllReduce` over device buffers. |
| `cuda_nccl_alltoall` | All-to-all personalized exchange (bisection bandwidth). | Grouped `ncclSend`/`ncclRecv` to every peer (NCCL has no native all-to-all). |
| `cuda_nccl_cg_step` | CG iteration skeleton (SpMV halo + two reductions). | Grouped `ncclSend`/`ncclRecv` halo + two `ncclAllReduce`. |
| `cuda_nccl_moe` | Top-1 MoE dispatch + combine with variable expert loads. | Two grouped `ncclSend`/`ncclRecv` phases with per-peer variable counts: dispatch followed by inverse combine. |

## Run

```bash
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_halo_1d 1048576 100 20
mpirun -np 2 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_pingpong 4194304 100 20
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_allreduce 4194304 100 20
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_alltoall 65536 100 20
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_cg_step 512 50 10
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_moe 16384 256 100 20 uniform,locality80,hotspot80
```
