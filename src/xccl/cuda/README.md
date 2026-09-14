# CUDA + NCCL

This backend uses NCCL for GPU-resident communication and MPI only for process
launch, NCCL bootstrap, and result collection.

## Implemented operations

| Benchmark | NCCL operation |
| --- | --- |
| `pingpong` | Matched point-to-point send and receive |
| `halo_1d` | Grouped neighbor sends and receives |
| `allreduce` | `ncclAllReduce` |
| `alltoall` | Grouped send and receive to every peer |
| `cg_step` | Grouped halo exchange and two allreduces |
| `moe` | Variable-count grouped dispatch and combine |

NCCL has no dedicated all-to-all or halo collective; grouped point-to-point
operations are its native expression of those patterns.

## Device API (`cuda_nccl_device`)

`device/` holds a second implementation of `alltoall` that issues its transfers
from inside the kernel, through NCCL's device API (`nccl_device.h`, NCCL 2.28 or
newer). The measured operation, buffer layout, validation and timing loop are
the same as `microbench/alltoall.cu`, so the pair isolates one variable: who
initiates the transfer.

| | `cuda_nccl` | `cuda_nccl_device` |
| --- | --- | --- |
| issues the transfer | host, per call | the kernel |
| per operation | grouped `ncclSend`/`ncclRecv` + launch | one kernel launch |
| buffers | `cudaMalloc` | `ncclMemAlloc` + symmetric window |

What that removes is the per-operation host enqueue, not the wire time. The
targets are built only when the NCCL found has a device API; against an older
one CMake reports that it is skipping them and the rest of the backend builds
normally.

Which GIN backend rings the NIC doorbell is a runtime choice and it matters:
`cluster/leonardo/runtime/nccl-device.sh` pins `NCCL_GIN_TYPE=2` (CPU proxy),
because NCCL selects GDAKI on its own there and GDAKI hangs on that machine.

## Build

Leonardo provides NCCL through NVHPC and uses the `leonardo-cuda-nccl` preset:

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-nccl
cmake --build --preset leonardo-cuda-nccl
```

For another cluster, provide `NCCL_INCLUDE_DIR` and `NCCL_LIBRARY` in a CMake
preset and register the resulting backend with the harness.

For benchmark semantics, see the
[benchmark contracts](../../../docs/README.md#benchmark-contracts). For
arguments, defaults, topologies, and launch examples, see the
[experiment operations](../../../cluster/harness/README.md#experiment-operations).
