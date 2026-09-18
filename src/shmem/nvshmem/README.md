# CUDA + NVSHMEM

This backend uses NVSHMEM symmetric GPU buffers and one-sided or team operations.

## Implemented operations

| Benchmark | NVSHMEM operation |
| --- | --- |
| `pingpong` | Persistent device-initiated puts and completion signals |
| `halo_1d` | Cooperative multi-block puts with neighbor signals |
| `allreduce` | `nvshmem_float_sum_reduce` |
| `alltoall` | `nvshmem_float_alltoall` |
| `cg_step` | Cooperative halo puts with neighbour signals, two one-element double reductions |
| `moe` | Multi-block puts with per-peer dispatch/combine signals, split per phase |

The ping-pong, halo, and CG-step implementations use cooperative kernels so
communication can be device initiated. On proxy-based inter-node transports
their grid size is capped by default, because each block there sends to a
*different* peer and every block's remote op is separately proxied.

`moe` is capped too, but at 64 rather than 8: its blocks split one peer's
payload, so the limit is how many concurrent proxied block-puts the host proxy
absorbs, not how many peers are in flight. A 2n1g sweep put the optimum at 64
(1656 / 877 / 2439 usec for uniform / locality80 / hotspot80) and measured the
full 992-block grid at roughly 1.9x slower with 50x the run-to-run variance.
`GPU_BENCH_NVSHMEM_MAX_BLOCKS` sets the cap for any of them.

## Device API (`cuda_nvshmem_device`)

`application/device/` holds a second implementation of `cg_step` that runs the
whole iteration inside one kernel and issues its own communication through
NVSHMEM's device API. The grid decomposition, stencil, dot products, signal
protocol, payload and validation are the same as `application/cg_step.cu`, so
the pair isolates one variable: who walks the iteration.

| | `cuda_nvshmem` | `cuda_nvshmem_device` |
| --- | --- | --- |
| halo | device-initiated puts + signals | same |
| pack / stencil / dots | four host-enqueued kernels | `__device__` phases, `grid.sync()` between |
| the two scalar sums | `nvshmemx_double_sum_reduce_on_stream` | `nvshmem_double_sum_reduce` in-kernel |
| per iteration | four launches, stream-ordered | none; the loop is resident |

`GPU_BENCH_CG_STEPS_PER_LAUNCH` (default 1) sets how many iterations one launch
runs. At 1 the loop is still host-walked and only the reduction path differs;
above 1 the loop is resident and the difference from 1 is what leaving the host
out of the iteration is worth.

Unlike `cuda_nccl_device` this needs no version guard: every NVSHMEM this
project builds against has device-side RMA, signals and team reductions.

Two things differ from `cuda_nvshmem` on purpose:

- `cluster/leonardo/runtime/nvshmem-device.sh` sets `NVSHMEM_DISABLE_NCCL=1`.
  There is no NCCL dispatch from inside a kernel, and aCG sets the same, so the
  recorded environment says what actually ran.
- `GPU_BENCH_NVSHMEM_MAX_BLOCKS` caps the *sending* blocks only. One grid does
  both communication and the stencil here, and the proxy's preference for few
  blocks must not become a cap on compute parallelism.

There is no phase pass. The phases are `grid.sync()`-separated regions of one
kernel, and splitting them apart to time them would just be the host-initiated
backend. aCG's device solver has the same shape and the same consequence: its
per-phase timers are all zero and the whole solve lands in `other`.

This backend exists to predict that solver (`--solver acg-device --comm
nvshmem`), which the host-initiated `cg_step` has no counterpart for.

## Build

Leonardo provides NVSHMEM through NVHPC and uses the
`leonardo-cuda-nvshmem` preset:

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-nvshmem
cmake --build --preset leonardo-cuda-nvshmem
```

The [halo analysis](../../../docs/analysis/halo-1d-methodology.md) explains the
device-initiated timing and transport considerations.

For benchmark semantics, see the
[benchmark contracts](../../../docs/README.md#benchmark-contracts). For
arguments, defaults, topologies, and launch examples, see the
[experiment operations](../../../cluster/harness/README.md#experiment-operations).
