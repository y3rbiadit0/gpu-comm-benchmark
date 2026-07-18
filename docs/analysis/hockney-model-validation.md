# Validating the Hockney α–β model against the Rico-Gallego survey

An assessment of the latency/bandwidth model used across this suite
(`T(m) = α + m/B∞`, knee `n½ = α·B∞` — see
[halo_1d-crossover.md](halo_1d-crossover.md) §2) against:

> J. A. Rico-Gallego, J. C. Díaz-Martín, R. R. Manumachu, A. L. Lastovetsky.
> *A Survey of Communication Performance Models for High-Performance
> Computing*. ACM Computing Surveys 51(6), Article 126, 2019.
> [doi:10.1145/3284358](https://doi.org/10.1145/3284358)

**Verdict: keep Hockney for this benchmark — the survey supports it for a
comm-only point-to-point characterization — but tune the fitting procedure.**
Switching to a richer model (LogGP, LogGPS, τ–Lop) would cost a lot and buy
almost nothing here; the real gaps are in *how the parameters are estimated*,
and the survey gives concrete, citable guidance on exactly that.

## 1. Why Hockney is the right model for this benchmark

The survey's criticisms of Hockney (§2.2.1, §5.2) target uses this suite does
not have:

- predicting **collectives** under ideal no-contention assumptions;
- predicting **applications** where CPU overhead vs. network latency matters
  for computation/communication overlap;
- **intra-node shared-memory** transfer chains — the motivation for the
  middleware models lognP, mlognP, and τ–Lop.

halo_1d is a comm-only point-to-point ring with fixed placement and nothing to
overlap, and the model is used *descriptively* — to characterize each backend
by (α, B∞, n½) and locate crossovers — not to predict application cost. That
is exactly the role the survey credits Hockney with: it is the model Thakur
et al. (2005) use for MPICH's runtime collective-algorithm selection.

Two structural points work in our favor:

1. **The LogP family's extra parameters (o, g) require host-CPU-centric
   measurement tricks** (e.g., LogGP's artificial-delay method for isolating
   the send overhead) that are undefined for device-initiated NVSHMEM puts or
   NCCL's kernel+proxy path. The survey stresses (§4, §5.1) that a model
   without a precise, reproducible measurement procedure "is unusable from a
   practical point of view" — adopting LogP/LogGP here would mean inventing
   one. Meanwhile the Nsight Systems timeline attribution
   ([halo_1d-crossover.md](halo_1d-crossover.md) §5) already decomposes α into
   launch / match / transfer / wait, which is the *insight* the o/L/g
   decomposition exists to provide.

2. **The halo ring is already a survey-blessed measurement experiment.** The
   survey cites Rico-Gallego's `Ring_τ` operation — a ring of concurrent
   `MPI_Sendrecv`s — as the way τ–Lop measures concurrent-transfer cost
   L(m, τ). The halo_1d ring is that experiment with τ = 2. So the fitted β is
   well-defined; it is the *concurrent-exchange* β, not the pingpong β.

## 2. What the survey says we should fix (tuning, in priority order)

### 2.1 A single (α, β) line over the whole sweep is explicitly warned against — fit piecewise

Survey Fig. 11 (p. 126:23) shows that narrow-range linear fits (small /
medium / large m) yield significantly different parameters, and Lastovetsky &
Rychkov's remedy is a **threshold parameter with separate equation sets per
regime** — the same rationale as PLogP making its parameters piecewise
functions of m.

Our own data already demands this:

- `cuda_nvshmem` has *three* regimes — latency floor → 33 GB/s plateau →
  16.1 GB/s cliff at ≥ 32 MB (single-block put collapse);
- the MPI backends will have an **eager → rendezvous** breakpoint (the entire
  motivation for LogGPS);
- NCCL switches protocols (LL / LL128 / Simple) with message size.

One Hockney line cannot represent a bandwidth *collapse*. The fix is not a new
model — it is per-segment (α, B∞) with named breakpoints ("32 MB =
single-block put collapse", "eager/rendezvous switch at UCX's threshold").

### 2.2 `characterize.py` doesn't match the doc, and neither matches the survey's method

[halo_1d-crossover.md](halo_1d-crossover.md) §2 says "linear-fit
`time_per_iter_s` against m: intercept = α, 1/slope = B∞", but
`tools/benchscribe/characterize.py` actually takes α = median observed latency
for messages up to 4 KiB and B∞ = peak observed bandwidth.

The survey's guidance (§4): compose **more than r points across a wide m
range, solve by least-squares regression, and report each parameter as
ā ± σā** (estimated value ± standard error). Small-message median and peak are fine as
sanity anchors, but the reported α / B∞ / n½ should come from a per-segment
regression with standard errors and R².

Also note that `n½ = α·B∞` currently multiplies a small-message floor by a peak
from another regime — per-segment regression fixes that inconsistency for free.

### 2.3 Report dispersion, not just means

Survey §5.1.2 (citing Hunold & Carpen-Amarie 2015/2016): don't assume
measurements are iid, use samples of size ≥ 30, report confidence intervals.
The harness's 100 iters / 20 warmup / 5 trials is adequate, but benchscribe
should surface per-point standard error so the quality of the fit can be
judged.

One thing the suite already does *right* per the survey: point-to-point
synchronization in the timed loop instead of `MPI_Barrier`, which Hunold &
Carpen-Amarie specifically flag as interfering with the benchmarked operation.

### 2.4 Document the "exchange model" caveat

The survey notes that all Hockney applications assume a node executes at most
one send and one receive simultaneously. The halo ring does 2 sends + 2
receives concurrently and accounts **bus bytes**, so the fitted α and β
describe the *full exchange*, not a single message. A sentence in the analysis
doc should say: **do not compare our α to published pingpong α**. In τ–Lop
terms, our β is measured under transfer concurrency τ = 2. This is also the
honest framing for the thesis.

## 3. When a different model *would* be needed

If the project later models **collectives** (e.g., the allreduce in the CG
step) or **application-level overlap**, Hockney's known failure modes kick in.
The survey's flagship example (§5.2): Scatter and Recursive-Doubling Allgather
get the *same* Hockney cost formula — `(α + mβ)·log₂P` — despite moving very
different data volumes. At that point the candidates are:

- **τ–Lop** (Rico-Gallego & Díaz-Martín 2015; 2016; 2017) — contention-aware,
  channel-aware, from the same authors as the survey; validated on
  heterogeneous CPU+GPU clusters over Ethernet and InfiniBand;
- **LogGP / LogGPS** — if send-overhead vs. network-latency separation or the
  eager/rendezvous synchronization cost must be modeled analytically rather
  than attributed via profiling.

For halo_1d as designed, they are overkill with an unsolvable measurement
problem on the GPU-initiated backends.

## 4. Action items

1. **Piecewise least-squares fit in `tools/benchscribe/characterize.py`**:
   segment the sweep at detected/known breakpoints, regress time-vs-bytes per
   segment, report α ± σ, B∞ ± σ, R², n½ per segment; keep min-floor / peak
   bandwidth as cross-checks.
2. **Per-point dispersion** (standard error across trials) in the benchscribe
   report.
3. **Caveats paragraph** in [halo_1d-crossover.md](halo_1d-crossover.md):
   exchange-model semantics (§2.4 above), breakpoints = protocol switches,
   and why not LogGP/τ–Lop — with the survey citations from this document.
