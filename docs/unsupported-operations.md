# Unsupported Operations Tracker

This document tracks communication capabilities that cannot currently be measured
directly, capabilities that still need runtime validation, and native operations
for which the suite uses an explicit alternative. Keep these categories separate:
an emulated or barrier-based benchmark is still runnable, while an unsupported
benchmark must report `status=NOT_IMPLEMENTED validation=SKIP` and must not emit a
timing.

Last updated: 2026-08-21.

## Confirmed Active-Suite Gaps

| Backend | Benchmark / operation | Affected topology | Current behavior | Evidence | Next iteration |
| --- | --- | --- | --- | --- | --- |
| `sycl_oneccl_oshmpi` | `allreduce` @ `2n4g` — **fixed 2026-08-25, needs re-run** | The benchmark completed and validated all 23 sizes, then crashed in `MPI_Finalize` with SIGBUS in `uct_mm_ep_flush`; the harness discarded it on the non-zero exit. | **Our bug, not the stack's.** `kvs`, `comm` and `stream` were declared at try-block scope in `src/xccl/sycl/*.cpp`, so they outlived `MPI_Finalize` and were destroyed after it. For the OSHMPI backend they hold shared-memory segments UCX still has endpoints into, so tearing MPI down first flushed into freed memory. Only `2n4g` fired it — the sole topology with intra-node (`mm`) and inter-node (`rc`) endpoints open together. Not a build mismatch: `ompi_info` shows Open MPI 4.1.6 was configured `--with-ucx=hpcx/2.18.1`, the UCX in the backtrace. | Fixed by scoping the oneCCL objects so they are released before `MPI_Finalize`, in `allreduce`, `alltoall`, `halo_1d`, `pingpong` and `cg_step`. `moe` is not scoped (its early NOT_IMPLEMENTED path finalizes mid-function) but has no OSHMPI variant, so the bug is latent there. Re-run allreduce for this cell. |
| `sycl_oneccl_oshmpi` | `cg_step`, grouped `ccl::send`/`ccl::recv` column-slab halo | `1n2g` confirmed; other topologies untested | Job hangs until the harness timeout kills it. Nothing can be emitted from inside the process — `moe.cpp`'s `NOT_IMPLEMENTED` path catches a backend *exception*, and a deadlock throws nothing. The `timeout` in `experiments/common.sh` is the only defense and it works: exit 124, no report line, no data recorded. | Leonardo 2026-08-21, `cg_step/sycl_oneccl_oshmpi/1n2g`: ran 2:01 at **0% CPU** (user time 0.02 s), killed at the 2 m timeout, output stopping after `CCL_BACKEND changed to be oshmpi`. **The NCCL-backed `sycl_oneccl` runs the identical benchmark and topology successfully** (2.0 s, `validation=PASS`, 150–154 µs across three trials), so this is specific to the OSHMPI backend, not to oneCCL grouped point-to-point in general. | Reduce to a standalone grouped-P2P reproducer on the OSHMPI backend and report upstream. Exclude `sycl_oneccl_oshmpi` from `cg_step` sweeps. |
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
| `sycl_oneccl` | `cg_step` halo using `ccl::send`/`ccl::recv` | **`1n2g` verified working 2026-08-21** (2.0 s, `validation=PASS`). Unexpected backend failures still abort the job. | Not yet safe to generalize. On `1n2g` cg_step's non-periodic line gives each rank a single peer, so the group holds **2 operations**; `halo_1d`'s periodic ring gives it **4**, and that is the case that stalls. On `1n4g` cg_step's interior ranks have two neighbours and post 4 grouped operations to 2 distinct peers — the same shape as the stalling `halo_1d` `1n4g` run. **Test `1n4g` before trusting cg_step on this backend.** |
| `sycl_oneccl` | `moe` variable-count P2P dispatch and combine | Recognized missing P2P support already emits `NOT_IMPLEMENTED`; unexpected hangs or errors remain failures. | Test one routing case on `1n2g` and `2n1g` before the full routing matrix. |
| `sycl_oneccl` | `alltoall` collective | A backend exception currently aborts the job. | Confirm whether the installed NCCL-backed fork implements `ccl::alltoall`; add explicit `NOT_IMPLEMENTED` output if absent. |
| `cuda_nvshmem` | Cooperative `halo_1d` kernel | Throws an error when cooperative launch is unavailable. | Leonardo A100 supports the feature; retain this check for other systems and classify unsupported hardware explicitly if portability is required. |

The NCCL-backed fork keeps `ccl_api_functions.cpp` backend-neutral and dispatches
groups through the existing `group_impl` layer. Its NCCL path uses native
`ncclGroupStart`/`ncclGroupEnd`, preserves outermost-group nesting, defers
per-operation completion, and publishes SYCL stream events only after NCCL has
enqueued the group. Grouped point-to-point is enabled by default for `halo_1d`,
multi-rank `cg_step`, and `moe`, and the `halo_1d` ring is validated on `1n2g`,
`1n4g`, `2n1g` and `2n4g`.

`sycl_oneccl` allreduce has no known capability gap. The installed point-to-point
implementation is also validated for two-endpoint pingpong on `1n2g` and `2n1g`.

## Native Gaps With Runnable Alternatives

These rows are not `NOT_IMPLEMENTED` benchmark results. The benchmark runs, but
its reported model must remain visible when interpreting performance.

| Backend | Missing or unsafe native operation | Implemented benchmark model | Comparability note |
| --- | --- | --- | --- |
| Open MPI (`cuda_mpi`, `sycl_mpi`) | With UCC disabled, `tuned`/libnbc reduce device buffers with the host `ompi_op`, staging device→host→device inside the collective. Large-message bandwidth then *falls* with size | `OMPI_MCA_coll_ucc_enable=1` is now the default in `cluster/leonardo/runtime/{mpi-cuda,sycl-mpi}.sh` | Measured 2026-08-20, allreduce 1n4g: 48655 µs → 351 µs at 16 MiB (**138.7×**), reaching 56% of NCCL. UCC is slightly worse below ~32 KiB (0.6–0.7× at 64 B and 4 KiB), so the crossover is ~32–64 KiB. **Any result produced before this change has a crippled `cuda_mpi` baseline and must be regenerated.** |
| `sycl_mpi` vs `cuda_mpi` | The two stacks link different MPI implementations — `hpcx-mpi/2.19` (CUDA env) vs stock `openmpi/4.1.6` (SYCL env) | Both are measured as-is; no attempt is made to force one MPI on both stacks | Intra-node the difference is invisible (pingpong 1n2g: 88.62 vs 88.88 GB/s, both at the NVLink ceiling). Inter-node it dominates (2n1g: 19.80 vs 12.07 GB/s — HPC-X's UCX drives multiple IB rails, stock Open MPI one). **Do not read `cuda_mpi` vs `sycl_mpi` as a CUDA-vs-SYCL result inter-node.** |
| Open MPI UCC (`cuda_mpi`, `sycl_mpi`) | UCC accelerates `MPI_Allreduce` but badly regresses `MPI_Alltoall`/`MPI_Alltoallv` | `OMPI_MCA_coll_ucc_enable=1` globally in `runtime/mpi-cuda.sh`, overridden to `0` in `experiments/{alltoall,moe}/common.sh` | Measured 2026-08-24 (job 53993204), `cuda_mpi` moe 1n4g GB/s: uniform 21.8→167.9 (7.7×), locality80 5.5→258.3 (**47×**), hotspot80 9.2→80.2 (8.7×) when UCC is turned off. With it off, `cuda_mpi` matches `sycl_mpi` to within 1% on every routing case — stock Open MPI has no working UCC and was running `tuned` throughout. Each backend is measured in its best configuration per operation; `print-env.sh` records the setting in every job log. **The moe and alltoall `cuda_mpi` columns collected before this change are invalid.** |
| NCCL | No native all-to-all collective | Grouped `ncclSend`/`ncclRecv` to every peer | Measures NCCL P2P emulation, not a collective implementation. |
| NCCL | No dedicated halo collective | Grouped neighbor `ncclSend`/`ncclRecv` | This is the native NCCL P2P expression of a halo exchange. |
| OSHMPI | Inter-node point-to-point flag waits can deadlock without target-side progress | Halo uses NBI puts + `quiet` + CUDA device sync + `shmem_barrier_all`; pingpong uses its barrier handshake | Halo timing includes device completion and a global barrier. |
| OSHMPI | No native device all-to-all is assumed | Per-peer one-sided `shmem_putmem` loop + barrier | Measures one-sided emulation. |
| OSHMPI | Reductions reach a host Open MPI op **when UCC is disabled**, and then segfault on device pointers (Leonardo jobs 53261883, 53263113). This was a *configuration* fault, not a library limitation — the same `OMPI_MCA_coll_ucc_enable=0` that crippled `cuda_mpi` | UCC is now on in `runtime/oshmpi.sh`, and `GPU_BENCH_OSHMPI_ALLREDUCE_MEM=device` (default) reduces on CUDA-space buffers. The binary refuses to run that path if it sees UCC disabled rather than crashing. `=staged` keeps the host-staging fallback | With UCC on (job 53263792) all 23 sizes validate and land within 1–3% of `cuda_mpi`, because both dispatch to the same UCC collective — OSHMPI's reduction *is* Open MPI's. The device path is 34× faster than staged at 16 MiB (354 µs vs 12160 µs, 1n4g). |
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
