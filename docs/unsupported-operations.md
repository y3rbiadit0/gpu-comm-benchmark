# Unsupported Operations Tracker

This document tracks communication capabilities that cannot currently be measured
directly, capabilities that still need runtime validation, and native operations
for which the suite uses an explicit alternative. Keep these categories separate:
an emulated or barrier-based benchmark is still runnable, while an unsupported
benchmark must report `status=NOT_IMPLEMENTED validation=SKIP` and must not emit a
timing.

Last updated: 2026-07-18.

## Confirmed Active-Suite Gaps

| Backend | Benchmark / operation | Affected topology | Current behavior | Evidence | Next iteration |
| --- | --- | --- | --- | --- | --- |
| `sycl_oneccl` | `halo_1d`, grouped `ccl::send`/`ccl::recv` ring | `1n2g`, `1n4g`, `2n4g` | Emit `NOT_IMPLEMENTED`, reason `intra_node_ring_point_to_point`, for every halo size and exit successfully. `2n1g` remains supported. | Leonardo smoke jobs 49728186, 49728187, and 49728190 stalled before the first report; their output contained only oneCCL/NCCL initialization. Job 49728189 (`2n1g`) completed in 12 seconds. | Reduce to a standalone grouped P2P reproducer; test separate streams, NCCL P2P transport controls, and a parity schedule; report the backend issue upstream. Do not substitute serialized timings in the current comparison. |

## Confirmed Primitive Gaps Outside The Active Suite

| Backend | Primitive | Suite impact | Policy |
| --- | --- | --- | --- |
| NCCL-backed oneCCL fork | `ccl::broadcast` | No active benchmark requires it; MPI handles oneCCL bootstrap broadcasts. | A future broadcast benchmark must report `NOT_IMPLEMENTED`; do not silently fall back to MPI for the measured operation. |

## Conditional Capabilities To Validate

These are not yet confirmed as unsupported. Run a short one-size, one-trial smoke
test before allocating a full matrix. If a backend rejects an operation, convert
that result to an explicit capability record rather than a timeout or generic
benchmark failure.

| Backend | Benchmark / operation | Current handling | Validation needed |
| --- | --- | --- | --- |
| `sycl_oneccl` | `cg_step` halo using `ccl::send`/`ccl::recv` | Unexpected backend failures abort the job. | Test `1n2g` and `2n1g` first. Its multi-operation P2P pattern may share the intra-node limitation observed by `halo_1d`. |
| `sycl_oneccl` | `moe` variable-count P2P dispatch and combine | Recognized missing P2P support already emits `NOT_IMPLEMENTED`; unexpected hangs or errors remain failures. | Test one routing case on `1n2g` and `2n1g` before the full routing matrix. |
| `sycl_oneccl` | `alltoall` collective | A backend exception currently aborts the job. | Confirm whether the installed NCCL-backed fork implements `ccl::alltoall`; add explicit `NOT_IMPLEMENTED` output if absent. |
| `cuda_nvshmem` | Cooperative `halo_1d` kernel | Throws an error when cooperative launch is unavailable. | Leonardo A100 supports the feature; retain this check for other systems and classify unsupported hardware explicitly if portability is required. |

`sycl_oneccl` allreduce has no known capability gap. The installed point-to-point
implementation is also validated for two-endpoint pingpong on `1n2g` and `2n1g`.

## Native Gaps With Runnable Alternatives

These rows are not `NOT_IMPLEMENTED` benchmark results. The benchmark runs, but
its reported model must remain visible when interpreting performance.

| Backend | Missing or unsafe native operation | Implemented benchmark model | Comparability note |
| --- | --- | --- | --- |
| NCCL | No native all-to-all collective | Grouped `ncclSend`/`ncclRecv` to every peer | Measures NCCL P2P emulation, not a collective implementation. |
| NCCL | No dedicated halo collective | Grouped neighbor `ncclSend`/`ncclRecv` | This is the native NCCL P2P expression of a halo exchange. |
| OSHMPI | Inter-node point-to-point flag waits can deadlock without target-side progress | Halo uses NBI puts + `quiet` + CUDA device sync + `shmem_barrier_all`; pingpong uses its barrier handshake | Halo timing includes device completion and a global barrier. |
| OSHMPI | No native device all-to-all is assumed | Per-peer one-sided `shmem_putmem` loop + barrier | Measures one-sided emulation. |
| OSHMPI | Device-resident allreduce is unavailable in the current path | Host-symmetric `shmem_*_sum_to_all` buffers | Includes device/host staging where required. |
| NVSHMEM | Host-callable neighbor signal wait is unavailable | Device-initiated signal waits for pingpong/halo; host-driven put + barrier for `cg_step` | The chosen progress model differs by benchmark and is part of the result. |
| NVSHMEM | Multiple proxied signal-add operations were unreliable with IBGDA disabled | Each active block completes its NBI puts before one monotonic signal per direction; inter-node grid capped by default | Supported workaround for the validated Leonardo proxy path. |

## Intentional Topology Exclusions

These are invalid benchmark shapes, not backend capability failures.

| Benchmark | Excluded topology | Reason |
| --- | --- | --- |
| `pingpong` | Any shape other than `1n2g` or `2n1g` | The benchmark requires exactly two endpoints. |
| `halo_1d` | `1n1g` | A periodic ring requires at least two ranks or PEs. |

## Reporting Rules

- Confirmed unavailable benchmark combinations emit `status=NOT_IMPLEMENTED`
  with `validation=SKIP`, zero timing fields, and a stable `reason` token.
- Benchscribe preserves those rows as `N/I` and excludes them from fits,
  bandwidth comparisons, deltas, and speedups.
- Backend errors, hangs, and validation failures are not `NOT_IMPLEMENTED` unless
  the missing capability is identified and reported consistently by every rank.
- Alternative implementations remain numeric results, with their communication
  model documented in output fields and benchmark notes.

## Next-Iteration Queue

1. Isolate the oneCCL intra-node grouped P2P ring failure and determine whether
   it belongs to oneCCL grouping, the NCCL backend, or the SYCL in-order stream.
2. Add non-hanging capability probes for oneCCL `cg_step` and `alltoall`.
3. Exercise oneCCL MoE across intra-node and inter-node topologies and extend its
   existing capability classification if the failure is topology-specific.
4. Decide whether OSHMPI barrier-based paths should remain in direct performance
   tables or move to a separate synchronization-model comparison.
5. Revalidate NVSHMEM with IBGDA enabled and remove proxy-path limits
   only when the alternate transport is confirmed.
