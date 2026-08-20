# halo_1d — 1D halo exchange benchmark

`halo_1d` is the suite's neighbor-communication benchmark. Every backend
implements the **same** experiment — a comm-only 1D halo exchange on a periodic
ring — so the six implementations differ only in *how* they move the halo, not
in *what* they measure. That makes them directly comparable.

This document explains the shared design once, then how each backend realizes
it, plus how to build, run, and read the output. For the performance modeling
(α–β latency/bandwidth fit and the one-sided-vs-two-sided crossover) see
[`docs/analysis/halo_1d-crossover.md`](analysis/halo_1d-crossover.md). For the
Leonardo job scripts see
[`cluster/leonardo/experiments/halo_1d/`](../cluster/leonardo/experiments/halo_1d/README.md).

---

## 1. What it measures

A **stencil halo exchange** is the communication pattern behind distributed
stencil solvers: each rank owns a contiguous slice of a 1D array and, to update
its slice, needs a thin border of values (the *halo* / *ghost cells*) from its
neighbors. `halo_1d` isolates **just that exchange** — no stencil math — so the
timing reflects the communication layer alone.

- **Topology:** a periodic ring. Rank `r` talks to `left = (r-1+P)%P` and
  `right = (r+1)%P`, so every rank has exactly two neighbors and all message
  counts are symmetric. Requires `P ≥ 2`.
- **Swept halo width `H`:** the per-message size in elements. The benchmark
  sweeps `H` (default: powers of two up to `max_halo`), so the run is also a
  message-size sweep.
- **GPU-resident buffers:** all sends/receives use device pointers — no host
  staging.
- **Backend-specific completion in the timed loop.** MPI, NCCL, oneCCL, and
  NVSHMEM use point-to-point completion. OSHMPI completes its two remote writes
  with `shmem_quiet`, closes CUDA-space work with `cudaDeviceSynchronize`, then
  calls `shmem_barrier_all`; the barrier guarantees progress for its passive RMA
  path, so OSHMPI timings include device synchronization and global barrier
  overhead.

---

## 2. Shared design

### Buffer layout (slice-local)

Each rank/PE allocates **one** buffer sized for the largest halo in the sweep
(`cap = max_halo`). It is *not* the whole global array — storage is `O(cap)` per
rank, not `O(N)`:

```
            cap            2*cap            cap
      |-----------|------------------|-----------|
buf:  [ left_halo |     interior     | right_halo]
                  ^
                  interior = buf + cap
```

- `interior` holds this rank's owned data; its two halves are tagged with
  distinct markers (below).
- `left_halo` / `right_halo` receive the neighbors' boundary values.

For a given halo width `H` the four regions are pure pointer arithmetic:

```
send_left  = interior                 // first H interior elems  (left boundary)
send_right = interior + n_local - H   // last  H interior elems  (right boundary)
recv_left  = interior - H             // left halo
recv_right = interior + n_local       // right halo      (n_local = 2*cap)
```

### The exchange

Each rank pushes its boundaries to its neighbors' halos:

```
my right boundary (send_right) ->  right neighbor's left  halo (recv_left)
my left  boundary (send_left)  ->  left  neighbor's right halo (recv_right)
```

In two-sided form (MPI/NCCL/oneCCL) that is two sends and two receives; in
one-sided form (NVSHMEM/OSHMPI) it is two remote writes, where symmetric
addressing means the destination pointer on the neighbor is the same
`recv_left` / `recv_right` we use locally — only the target PE differs.

### Validation (local, no gather)

The interior's first half is tagged `2*(rank+1)` ("left boundary" marker) and
its second half `2*(rank+1)+1` ("right boundary" marker) — both exactly
representable in float. After the exchange each rank checks its received halos
**against what its neighbors must have sent**, with no gather to a root:

```
recv_left  must equal  2*(left+1)+1     // left neighbor's right boundary
recv_right must equal  2*(right+1)       // right neighbor's left boundary
```

This also verifies orientation (a left/right swap would be caught), and scales
to thousands of ranks without the float losing exactness. The per-rank result
is combined with an AND-reduction across ranks. Receive halos are poisoned
before and validated after every measured batch, outside the timed interval.

### Bytes and bandwidth convention

Reported `gbytes_per_s` is **bus** (send+receive) bandwidth — both sends and
both receives per rank per iteration, applied uniformly across all backends:

```
bytes_per_iter = 4 * H * sizeof(float)            (sendL + sendR + recvL + recvR)
gbytes_per_s   = bytes_per_iter / time_per_iter_s
```

(The `extra` field records `bw=sendrecv` so this is unambiguous. Don't compare
these numbers against a vendor's unidirectional figure.)

### Timing

The shared batch harness (`include/timing.hpp`) reports two cases for every
`H`: `isolated` uses one exchange per batch, while `steady` uses `iterations`
exchanges per batch. A timed sample is the completed batch duration divided by
its exchange count. Each case records `GPU_BENCH_BATCH_SAMPLES` samples
(default `10`), reduced across ranks sample by sample with MAX, so
`time_per_iter_s` remains the **average slowest-rank amortized exchange**.
`min`/`max` and the quartiles describe that same reduced sample series. Only
rank/PE 0 prints.

---

## 3. CLI and output

All six binaries share one interface:

```
<binary> <max_halo_elems> <iterations> <warmup> [comma-separated halo sizes]
```

| Arg | Meaning | Default |
| --- | --- | --- |
| `max_halo_elems` | largest halo width; sizes the allocation | `1048576` (1 Mi elems = 4 MiB) |
| `iterations` | exchanges per steady-state batch | `100` |
| `warmup` | untimed iterations per halo width | `20` |
| `halo sizes` | explicit comma-separated sweep | powers of two up to `max_halo_elems` |

Two lines are printed per swept `H` (the standard `print_report` format, which
`tools/benchscribe` parses):

```
cuda_nvshmem_halo_1d n=1024 ranks=4 bytes=16384 iters=1 warmup=20 \
  time_per_iter_s=4.2e-06 usec=4.2 gbytes_per_s=3.90 case=isolated \
  timing=batch batch_iters=1 batch_samples=10 validation=PASS
cuda_nvshmem_halo_1d n=1024 ranks=4 bytes=16384 iters=100 warmup=20 \
  time_per_iter_s=3.1e-06 usec=3.1 gbytes_per_s=5.28 case=steady \
  timing=batch batch_iters=100 batch_samples=10 validation=PASS
```

`n` is the halo width `H`; `bytes` is `4*H*sizeof(float)`; the `extra` tail
(`halo_elems`, `topology`, `bw`, and a per-backend `sync=`/`device=`) documents
the configuration.

---

## 4. The six implementations

Same benchmark, different transport. The contrast between **host-initiated**
two-sided, **host-initiated** one-sided, and **device-initiated** one-sided is
the whole point.

| Target | Source | Transport / sync | Init |
| --- | --- | --- | --- |
| `cuda_mpi_halo_1d` | `src/mpi/cuda/halo_1d.cu` | Persistent CUDA-aware MPI requests | host two-sided |
| `cuda_nccl_halo_1d` | `src/xccl/cuda/halo_1d.cu` | grouped `ncclSend`/`ncclRecv` on a stream | host two-sided |
| `cuda_nvshmem_halo_1d` | `src/shmem/nvshmem/halo_1d.cu` | Persistent cooperative multi-block puts + signals | **device** one-sided |
| `oshmpi_halo_1d` | `src/shmem/oshmpi/halo_1d.cu` | NBI puts + `quiet` + CUDA sync + barrier | host one-sided |
| `sycl_mpi_halo_1d` | `src/mpi/sycl/halo_1d.cpp` | Persistent SYCL-aware MPI requests (USM) | host two-sided |
| `sycl_oneccl_halo_1d` | `src/xccl/sycl/halo_1d.cpp` | point-to-point `ccl::send`/`ccl::recv` (events) | host two-sided |

### cuda_mpi / sycl_mpi — two-sided MPI

Create two persistent receives and two persistent sends for each halo width.
Every exchange calls `MPI_Startall` and `MPI_Waitall`; requests are reused for
warmup and both measured cases. Tags separate rightward (`0`) from leftward
(`1`) messages even at `P = 2`. Pointers are device (CUDA) / USM (SYCL), so
these test **CUDA-aware / SYCL-aware** MPI, not host staging.

### cuda_nccl — grouped point-to-point

NCCL has no halo collective, so the exchange is modeled with
`ncclGroupStart()` ... two `ncclRecv` + two `ncclSend` ... `ncclGroupEnd()` issued
on a CUDA stream. A batch queues that group repeatedly, then calls
`cudaStreamSynchronize` once. NCCL matches send/recv per peer by posting order.
MPI is used only to bootstrap NCCL and reduce the timing result.

### cuda_nvshmem — device-initiated, signal sync

Each batch runs inside one persistent cooperative kernel. Multiple blocks split
the halo into chunks and issue cooperative NBI puts; after every active block
completes its operations, block 0 publishes one signal in each direction and
waits for the matching incoming signals. A grid synchronization carries the
dependency into the next exchange. Signal values are monotonic and there is no
global barrier inside the batch. The target uses separable CUDA compilation for
device-side NVSHMEM calls.

### oshmpi — host-initiated one-sided, barrier completion

Two `shmem_putmem_nbi`s write the boundaries into the neighbors' halos (on
device-symmetric memory from the OSHMPI CUDA memory space). `shmem_quiet`
completes the OpenSHMEM operations, then `cudaDeviceSynchronize` closes
CUDA-space work that OSHMPI may have enqueued. `shmem_barrier_all` confirms that
every PE has reached the completion boundary. Point-to-point flag waits are not
used because passive RMA can require target-side progress on inter-node paths.
The timed loop therefore includes device synchronization and one global barrier
per exchange. Per-PE timings are reduced to PE 0 through the symmetric heap
(there is no direct MPI use here). Launch with `oshrun`.

### sycl_oneccl — oneCCL point-to-point

Models the ring with oneCCL `ccl::recv`/`ccl::send` (the natural analog of
`ncclSend`/`ncclRecv`): group both receives and both sends with
`ccl::group_start()`/`ccl::group_end()`. A batch queues every group before
waiting on its events. Grouping avoids an in-order stream deadlock. **Caveat:**
not every oneCCL build implements point-to-point send/recv (the UNISA
NCCL-backed fork in particular); if they are missing this binary reports a
backend error instead of results. See the
[`src/xccl/sycl/README.md`](../src/xccl/sycl/README.md) caveat.

The Leonardo NCCL-backed fork dispatches public groups through `group_impl` to
native NCCL groups and defers event publication until the outermost group ends.
Grouped point-to-point is enabled by default and the ring is validated on
`1n2g`, `1n4g`, `2n1g`, and `2n4g`.

---

## 5. Build & run

Each backend is a CMake preset, driven through the top-level `Makefile`:

```bash
make build PRESET=leonardo-cuda-mpi       # also: leonardo-cuda-nccl,
make build PRESET=leonardo-cuda-nvshmem   #       leonardo-oshmpi,
make build PRESET=leonardo-sycl-mpi       #       leonardo-sycl-oneccl
```

Run with 4 ranks/PEs (one GPU per rank):

```bash
# two-sided MPI (CUDA / SYCL)
mpirun -np 4 ./build/leonardo-cuda-mpi/src/mpi/cuda/cuda_mpi_halo_1d        1048576 100 20
mpirun -np 4 ./build/leonardo-sycl-mpi/src/mpi/sycl/sycl_mpi_halo_1d        1048576 100 20

# NCCL
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_halo_1d     1048576 100 20

# NVSHMEM (device-initiated)
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_halo_1d 1048576 100 20

# OSHMPI (launch with oshrun)
oshrun  -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_halo_1d       1048576 100 20

# oneCCL (see caveat)
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_halo_1d 1048576 100 20
```

Sweep just a few sizes, or change trial counts, via the optional args:

```bash
# H ∈ {1, 16, 256, 4096}, 500 timed iters, 50 warmup
mpirun -np 4 ./build/leonardo-cuda-mpi/src/mpi/cuda/cuda_mpi_halo_1d 4096 500 50 1,16,256,4096
```

On Leonardo, the SLURM job scripts under
[`cluster/leonardo/experiments/halo_1d/`](../cluster/leonardo/experiments/halo_1d/README.md)
wrap these (including an `nsys` profiling mode).

---

## 6. Notes & caveats

- **`P ≥ 2`** is required (a ring needs at least two PEs); the binaries error out
  otherwise.
- **Allocation is `O(max_halo)` per rank**, not `O(N)`. The benchmark is
  comm-only, so the owned interior is just large enough to source disjoint
  send-left / send-right regions for every swept `H`.
- **Bus bandwidth**: `gbytes_per_s` counts 4 messages/rank/iteration. Keep that
  in mind when comparing to unidirectional link figures.
- **Not all backends are strict equals.** `cuda_mpi` vs `sycl_mpi` is the
  closest fair pair because both use the same two-sided algorithm.
  `cuda_nvshmem` and `oshmpi` both use one-sided remote writes, but NVSHMEM uses
  neighbor signals while OSHMPI uses a global barrier, so their synchronization
  costs differ. NCCL/oneCCL are point-to-point over collective libraries.
- These sources are validated on Leonardo (A100); they are not built locally.
  The α–β model and crossover analysis live in
  [`docs/analysis/halo_1d-crossover.md`](analysis/halo_1d-crossover.md).
