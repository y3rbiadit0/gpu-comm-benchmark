# cg_step Cross-Backend Revision

Fairness and idiom review of the six `cg_step` implementations
(`src/{mpi,xccl}/{cuda,sycl}`, `src/shmem/{nvshmem,oshmpi}`), verifying that
(a) the compute side is a controlled variable, (b) each backend uses its
communication library idiomatically, and (c) the remaining asymmetries are
explainable library properties rather than implementation accidents.

**Verdict:** the six backends are structurally fair and each uses its library
idiomatically. No correctness problems found. Five asymmetries need to be
explainable in the results chapter — four are justified library properties to
document, one is a small avoidable unfairness against oneCCL.

## What is identical everywhere (the fairness foundation)

The timed region has the same shape in every backend:

> pack 2 boundary columns → neighbor exchange → unpack into ghost columns →
> 5-point stencil → local dot(p,q)/dot(q,q) → two scalar allreduces

Beyond the shape:

- The four CUDA backends (`cuda_mpi`, `cuda_nccl`, `cuda_nvshmem`, `oshmpi`)
  share **textually identical kernels** (`init_p_kernel`, `pack/unpack_column_kernel`,
  `spmv_kernel`, `cg_dot_kernel` with the shared-memory block reduction) and
  identical launch configurations (16×16 2D blocks, 256-thread 1D blocks, dot
  grid capped at 4096 blocks). Compute is a controlled variable; differences
  between them are pure communication.
- The two SYCL backends are likewise identical to each other.
- Ring mapping is consistent across all six (`send_west → left` lands in the
  left neighbor's `recv_east`, and symmetrically for east).
- Validation is identical: exact brute-force references for dot(p,q)=S(S−1)
  and dot(q,q), plus per-rank `validate_columns` on q (which proves the halo
  arrived correctly). All local, no gather.
- `bytes_per_iter = 2·side·sizeof(float)` (halo volume; the reductions are
  scalars) in all six reports.
- Timing aggregation follows the suite convention: `MPI_Reduce` of
  avg(max)/min/max for five backends, manual symmetric-heap gather to PE 0 for
  OSHMPI.

## Per-backend idiom check

| Backend | Halo exchange | Reductions | Assessment |
| --- | --- | --- | --- |
| `cuda_mpi` | 2× blocking CUDA-aware `MPI_Sendrecv` on packed device columns | 2× `MPI_Allreduce` on device scalars | ✅ canonical host-driven baseline |
| `cuda_nccl` | grouped `ncclSend`/`ncclRecv` (`ncclGroupStart/End`) | 2× `ncclAllReduce` | ✅ best-in-class use of the library: the whole iteration is enqueued on one stream with a single `cudaStreamSynchronize` at the end, exploiting NCCL's stream-ordered semantics |
| `cuda_nvshmem` | host `nvshmem_float_put` + `nvshmem_quiet` + `nvshmem_barrier_all` | `nvshmem_double_sum_reduce` team collective on device symmetric scalars | ✅ with the documented host-driven constraint (asymmetry 1) |
| `oshmpi` | `shmem_putmem` (CUDA symmetric space) + `shmem_quiet` + `shmem_barrier_all` | host-staged `shmem_double_sum_to_all`, alternating psync/pwrk pairs | ✅ correct SHMEM idiom; psync reuse across iterations is safe because the two collectives alternate arrays |
| `sycl_mpi` | mirrors `cuda_mpi` (device-pointer `MPI_Sendrecv`) | mirrors it | ✅ |
| `sycl_oneccl` | `ccl::send`/`ccl::recv`, all four events collected then waited | 2× `ccl::allreduce(...).wait()` | ✅ modulo the UNISA fork's pt2pt caveat (documented in the source header) |

## Asymmetries to document (or fix)

### 1. One-sided backends pay a global barrier per halo — the big one

MPI/NCCL/oneCCL synchronize *only with the two neighbors*; NVSHMEM and OSHMPI
use `put + barrier_all`, an O(log P) **global** synchronization per iteration.
This is forced, not sloppy:

- host-callable `nvshmem_signal_wait_until` does not exist (point-to-point
  signal waits are device-only in NVSHMEM), so a host-driven put has no
  per-neighbor completion primitive;
- OSHMPI's spin-wait on a flag deadlocks inter-node without target-side
  progress (the pingpong lesson), so the barrier is the reliable sync.

Consequence: at scale these backends measure "halo + global barrier", not
"halo". Document as a property of *host-driven* one-sided models; the
device-initiated NVSHMEM `halo_1d` variant (put_signal + signal_wait in-kernel)
shows what the idiom achieves without the barrier.

### 2. OSHMPI stages the reduction scalars through the host

Two blocking `cudaMemcpy` D2H inside the timed loop
(`src/shmem/oshmpi/cg_step.cu:222-223`) that no other backend has, because the
reduction source/destination live on the **host** symmetric heap — a deliberate
property of the SHMEM-over-MPI model. Keep it; flag its numbers as not directly
comparable in every results table.

### 3. NCCL is the only backend with one host sync per iteration

All others round-trip to the host between pack, exchange, and compute. This is
a genuine library property (stream-ordered semantics is *the* NCCL advantage),
so it is fair — but part of any NCCL win is launch-overhead hiding rather than
transport speed. An Nsight timeline showing the host-sync gaps in the other
backends is the supporting figure if needed.

### 4. CUDA-vs-SYCL comparisons carry compute confounds

The SYCL dot uses `sycl::reduction` (runtime-chosen strategy) versus the
hand-written CUDA block reduction, and SYCL packs/unpacks both columns in one
kernel where CUDA launches two. Within-CUDA and within-SYCL comparisons are
clean; cross-language ones should be framed as **stack** comparisons
(compiler + runtime + comm library), not comm-library comparisons.

### 5. oneCCL's serialized allreduces — the one avoidable unfairness

`src/xccl/sycl/cg_step.cpp:157-158`: each `ccl::allreduce(...).wait()` forces a
host sync *between* the two reductions, while NCCL enqueues both and syncs
once. Fix: collect both events, wait after both are submitted (matching how
the same file already batches the four halo events). Effect is small (two
scalar allreduces) but it is the only place a backend is prevented from
pipelining that its API allows. **Status: open.**

## Cosmetic (not fairness) notes

- `nvshmem_quiet()` / `shmem_quiet()` immediately before `barrier_all` is
  redundant — barrier semantics subsume quiet. Harmless; remove or comment for
  explainability.
- In-loop zeroing of the partials uses blocking `cudaMemset` in the host-driven
  CUDA backends vs. `cudaMemsetAsync` in NCCL — consistent with each backend's
  sync model, no action needed.

## Bottom line

The comparison is defensible as designed. Asymmetries 1–4 are precisely the
findings a comm-library comparison exists to surface — they belong in the
caveats table of the results chapter, not engineered away. The only code change
worth making is the oneCCL event batching (asymmetry 5).
