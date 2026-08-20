# halo_1d: where one-sided beats two-sided, and why

A latency/bandwidth analysis of the comm-only 1D halo exchange across six GPU
communication backends on Leonardo (A100, `boost_usr_prod`). The goal is not a
single "fastest" verdict but a **model** that explains *which regime each backend
wins and where the crossover sits* — and a method to back that model with
Nsight Systems timelines instead of just wall-clock numbers.

> Status: the model and method below are complete; the result tables are
> templates to fill from a clean Leonardo run (`tools/benchscribe`) plus the
> profiled run described in [Profiling](#profiling-with-nsight-systems). The
> historical ratios quoted as motivation predate the comm-only rewrite and are
> labelled as such — regenerate before citing.

## 1. What is measured

Each backend runs the same kernel: a **periodic ring** where rank `r` exchanges a
halo of width `H` with `left=(r-1+P)%P` and `right=(r+1)%P`, using GPU-resident
buffers. Synchronisation in the timed loop is point-to-point for the MPI, NCCL,
oneCCL, and NVSHMEM implementations; OSHMPI instead completes each **batch** with
`shmem_quiet`, CUDA device synchronization, and a global `shmem_barrier_all`.
`H` is swept; for each
`H` the harness reduces the per-iteration times across ranks with MAX and
reports the mean of that series via `print_report`.

The bytes accounted per iteration are **bus bytes** — both sends and both
receives:

```
bytes_per_iter = 4 * H * sizeof(float) = 16 * H        (sendL + sendR + recvL + recvR)
gbytes_per_s   = bytes_per_iter / time_per_iter_s      (bus bandwidth)
```

So the reported `gbytes_per_s` is bus bandwidth, and the H-sweep is exactly a
message-size sweep — which is what makes the α–β fit below possible.

## 2. The model: α–β, and the communication roofline

Model the per-iteration time as a fixed latency plus a transfer term:

```
T(m) = α + m / B∞
```

- `m = bytes_per_iter = 16·H` — the message size knob.
- `α` (seconds) — the **latency floor**: everything that does not scale with
  message size (kernel launch, MPI matching, proxy-thread handoff, signal
  handshake, network RTT).
- `B∞` (bytes/s) — the **asymptotic bus bandwidth**: the slope ceiling as
  `m → ∞`.

This is the communication analogue of the roofline. Two regimes:

| Regime | Condition | Behaviour | Limited by |
| --- | --- | --- | --- |
| Latency-bound | `m ≪ α·B∞` | `T ≈ α` (flat) | fixed overhead → **α** |
| Bandwidth-bound | `m ≫ α·B∞` | `T ≈ m/B∞` (linear) | link/protocol → **B∞** |

The knee — the message size where the two terms are equal — is the
**half-bandwidth point**:

```
n½ = α · B∞          (m at which you reach 50% of B∞)
```

`n½` is the single most useful number to compare backends: a backend with a tiny
`α` reaches peak bandwidth at a small message, i.e. it is good at *exactly the
small, frequent halos that real stencil codes send*. Two backends can share the
same `B∞` and still differ by 10× at `H = 16` purely because of `α`.

**How to get α and B∞ from the data:** linear-fit against `m = 16·H` across the
sweep. Intercept = `α`, `1/slope = B∞`. Do it per `(backend, topology)`.
(`tools/benchscribe` gives the per-H rows; the fit is a two-line least-squares —
add it there if you want it automated.)

Fit **`min_usec`, not `usec`**, for the `isolated` case. Each isolated sample is
one exchange preceded by a barrier, and barrier-exit skew is strictly additive:
it inflates the mean but leaves the floor alone. The skew is also
backend-dependent (`MPI_Barrier` vs `nvshmem_barrier_all` vs
`shmem_barrier_all`), so fitting `usec` compares three barrier implementations
as much as three transports. `min_usec` is the minimum of the same MAX-reduced
series, which is the cleanest available estimator of the latency floor. Use
`usec` for the `steady` case, where the batch already amortizes the skew.

## 3. Hardware ceilings to anchor B∞ (confirm on node)

Fill these from a pingpong / `nvidia-smi topo -m` measurement; do not trust
spec sheets. Rough anchors for Leonardo Booster (4× A100-64 per node):

| Path | Transport in this suite | Order-of-magnitude B∞ to expect |
| --- | --- | --- |
| Intra-node (1n4g) | NVLink + CUDA IPC | hundreds of GB/s per peer |
| Inter-node (2n4g) | InfiniBand (`ibrc`/UCX) | ~tens of GB/s per peer |

The point of measuring `B∞` is to report **% of peak**, not raw GB/s — that is
what turns "NVSHMEM did 180 GB/s" into "NVSHMEM hit 85% of the NVLink ceiling
while NCCL hit 30%."

## 4. Per-backend hypotheses (what the model predicts)

| Backend | Mechanism | Predicted α | Predicted B∞ | Where it wins |
| --- | --- | --- | --- | --- |
| `cuda_nvshmem` | persistent cooperative multi-block puts, in-kernel signal wait | **lowest steady-state** — no host or MPI match per exchange | high intra-node; proxy-limited inter-node without IBGDA | small H and large intra-node halos |
| `oshmpi` | host one-sided NBI puts, `quiet` + CUDA sync + global barrier once per batch | low–moderate | moderate | small H, when device-kernel issue isn't available |
| `cuda_mpi` | CUDA-aware `Isend`/`Irecv`/`Waitall` | moderate (host + UCX) | good | the baseline; large H |
| `sycl_mpi` | SYCL-aware `Isend`/`Irecv` | ≈ `cuda_mpi` | ≈ `cuda_mpi` | sanity check vs `cuda_mpi` |
| `cuda_nccl` | grouped `ncclSend`/`ncclRecv` | **high** — kernel launch + proxy thread per exchange | high once amortised | only large H |
| `sycl_oneccl` | `ccl::send`/`ccl::recv` P2P (fork caveat) | n/a — not a fair P2P comparator | n/a | correctness/coverage only |

The headline crossover story to validate:

1. **Small H is an α contest.** Compare the `isolated` case to expose submission
   and completion latency, then `steady` to measure how much persistent or queued
   work amortizes it. NVSHMEM pays one cooperative launch per batch and performs
   each exchange in-kernel without tag matching. Treat all pre-batch numbers as
   historical and re-measure the ordering for both cases.

2. **Going inter-node widens the one-sided lead.** Crossing to IB raises every
   `α` (network RTT) and lowers every `B∞`, but the host-mediated backends pay
   that cost *on the host critical path*, while the device-initiated NVSHMEM put
   overlaps issue and transfer. Prediction: NVSHMEM's relative advantage grows
   from 1n4g → 2n4g (the old table moved 15× → 21×). Confirm, and explain via the
   `α` gap from the fit.

3. **Large H tests transport saturation.** NVSHMEM now splits the payload across
   a cooperative grid, removing the former single-block ceiling. The remaining
   crossover question is whether NCCL/MPI saturate the inter-node transport more
   effectively than NVSHMEM's proxy path when IBGDA is disabled.

## 5. Profiling with Nsight Systems

A wall-clock number tells you *that* NVSHMEM is faster; the timeline tells you
*why*, and lets you attribute `α` to launch vs match vs transfer vs wait.

Run a dedicated, single-trial, profiled job (profiling perturbs timing — never
report numbers from it):

```bash
GPU_BENCH_PROFILE=nsys GPU_BENCH_NTRIALS=1 \
  tools/sbatch.sh cluster/leonardo/experiments/halo_1d/cuda_nvshmem/2n4g.sh
GPU_BENCH_PROFILE=nsys GPU_BENCH_NTRIALS=1 \
  tools/sbatch.sh cluster/leonardo/experiments/halo_1d/cuda_mpi/2n4g.sh
GPU_BENCH_PROFILE=nsys GPU_BENCH_NTRIALS=1 \
  tools/sbatch.sh cluster/leonardo/experiments/halo_1d/cuda_nccl/2n4g.sh
```

This writes one report per rank under
`results/<name>/halo_1d/profiles/<job>-<id>-<trial>-rank<N>.nsys-rep`.
Open in the Nsight Systems GUI, or summarise headless:

```bash
nsys stats --report cuda_gpu_kern_sum,nvtx_sum,mpi_event_sum <file>.nsys-rep
```

Knobs: `GPU_BENCH_NSYS_TRACE` (default `cuda,nvtx,mpi`; add `ucx` to see the IB path on
2-node runs). To cut overhead you can profile a single rank by gating on
`SLURM_PROCID` in the rank wrapper — all-ranks is the default.

### What to look for, per backend

| Backend | On the timeline | `α` is dominated by |
| --- | --- | --- |
| `cuda_nvshmem` | one persistent cooperative kernel per batch; **no host MPI in the hot loop**; puts + `signal_wait` live inside the kernel | isolated: launch + signal handshake; steady: in-kernel exchange dependency |
| `cuda_mpi` | host `MPI_Startall/Waitall` on persistent requests, transfers on the UCX row, device buffers | host issue + UCX transfer + the `Waitall` stall |
| `cuda_nccl` | an `ncclDevKernel` on the CUDA row **plus** a proxy-progress thread on a CPU row | kernel launch *and* proxy handoff — the gap before the kernel runs is the tell |

Concretely, for each backend at a fixed small `H`: measure the per-iteration
critical path on the timeline, decompose it (launch gap / transfer / wait), and
check it against the fitted `α`. They should agree to within harness overhead —
when they don't, the timeline usually reveals a serialisation the wall clock hid.

## 6. Results

### 6.0 Historical single-block finding

These results predate the canonical persistent multi-block implementation and
explain why it replaced the single-block design:

- **α moves with the fabric, but only a little.** 13.6 µs (NVLink) → 16.7 µs (IB):
  the network adds just **~3.1 µs**. So ~80% of the latency floor is kernel launch +
  signal handshake, not the link — this is why device-initiated NVSHMEM is a
  latency king.
- **B∞ does not move at all.** Peak bus bandwidth is ~33 GB/s on *both* fabrics,
  even though NVLink offers far more than that. The ceiling is therefore the
  single-block `put_signal_nbi_block` issue throughput, **not** the interconnect.
- **The large-message collapse does not move either.** Both fall to the same
  ~16.1 GB/s plateau at the same ≥32 MB threshold — disproving an IB-saturation
  explanation (the intra-node run shows the identical cliff with no IB at all).

The canonical implementation applies that multi-block design. Re-run both batch
cases before using the following historical values in a current comparison.

### 6.1 Latency / bandwidth (α from small-message floor, B∞ from peak bus BW)

| Backend | Topology | α (µs) | B∞ (GB/s) | n½ (≈) | large-msg plateau |
| --- | --- | ---: | ---: | ---: | ---: |
| `cuda_nvshmem` | 1n4g | 13.6 (min 12.8) | 33.0 @16 MB | ~439 KB | 16.1 GB/s |
| `cuda_nvshmem` | 2n4g | 16.7 (min 13.4) | 33.0 @16 MB | ~539 KB | 16.1 GB/s |
| `oshmpi` | 1n4g | | | | |
| `cuda_mpi` | 1n4g / 2n4g | | | | |
| `cuda_nccl` | 1n4g / 2n4g | | | | |

> α = median per-iteration time for messages up to 4 KiB; B∞ = peak
> `gbytes_per_s`; n½ = α·B∞. Regenerate this table from raw results with
> `python3 tools/benchscribe results/ --benchmark halo_1d --fit`. Sweep:
> `GPU_BENCH_N=16777216`, 100 iters / 20 warmup.

### 6.2 Crossover

| Comparison | Topology | Crossover H (elems) | Crossover m (bytes) | Notes |
| --- | --- | ---: | ---: | --- |
| NVSHMEM vs NCCL (bandwidth) | 1n4g | | | NCCL overtakes above this H? |
| NVSHMEM vs CUDA-aware MPI | 2n4g | | | |

### 6.3 Timeline attribution (small H)

| Backend | Topology | Critical path (µs) | launch | transfer | wait | matches fitted α? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `cuda_nvshmem` | 2n4g | | | | | |
| `cuda_mpi` | 2n4g | | | | | |
| `cuda_nccl` | 2n4g | | | | | |

## 7. Reproduce

```bash
# clean timing sweep (numbers to report)
GPU_BENCH_NTRIALS=5 tools/sbatch.sh cluster/leonardo/experiments/halo_1d/cuda_nvshmem/2n4g.sh
# ... per backend/topology, then:
python3 tools/benchscribe results/ --benchmark halo_1d        # normalise vs cuda_mpi
python3 tools/benchscribe results/ --benchmark halo_1d --fit  # α, B∞, n½ per backend

# timeline (separate, not reported)
GPU_BENCH_PROFILE=nsys GPU_BENCH_NTRIALS=1 sbatch .../cuda_nvshmem/2n4g.sh
```

Fair-comparison caveats from the experiment README still apply: the MPI variants
are host-mediated baselines, and `sycl_oneccl` is included for coverage, not as a
P2P performance competitor. The cleanest apples-to-apples pairs are
`cuda_nvshmem` vs `oshmpi` (both one-sided) and `cuda_mpi` vs `sycl_mpi`.
