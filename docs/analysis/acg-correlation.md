# Correlation with aCG on Leonardo

This analysis tests how well the communication benchmarks predict the measured
performance of aCG on Leonardo. Every stored OSHMPI application result predates
the experimental staged-reduction source change and uses CUDA-aware
`MPI_Allreduce` for scalar reductions. The current worktree defaults to staged
`shmem_double_sum_to_all`, but that implementation has not been built or run on
Leonardo and is not evidence for this analysis.

## Data

- aCG application results: `aCG-native/acg-results/`
- archived OSHMPI halo with MPI reduction:
  `aCG-native/acg-results/acg-cg-oshmpi-backup-mpi/`
- microbenchmarks: `docs/analysis/data/1. microbenchmarks/`
- CG-step benchmark: `docs/analysis/data/2. application_benchmark/cg_step/`

All data were collected on Leonardo with one MPI process per A100 GPU. The
matched production topologies are `1n2g`, `1n4g`, `2n4g`, `4n4g`, and `8n4g`,
or 2, 4, 8, 16, and 32 processes.

The aCG comparison selects the fastest valid solver trial in each matrix,
backend, and topology cell, following aCG's paper reporting method. Total solver
time is divided by the iteration count for application-step comparisons. For
the reduction comparison, the measured aCG `allreduce` microseconds per
operation are compared with half of the `cg_step` reduce phase, because each
benchmark step performs two one-double reductions.

The error metric is

```text
MAPE = mean(abs(predicted / observed - 1))
```

over the five matched topologies. Pearson correlation measures whether the
topology trend has the same shape. It does not establish calibration: a result
can have high correlation and still have large absolute error.

## Backend mapping

| aCG result | Benchmark backend | Reason |
| --- | --- | --- |
| `acg-cg-mpi` | `cuda_mpi` | MPI halo and CUDA-aware `MPI_Allreduce` |
| `acg-cg-nccl` | `cuda_nccl` | NCCL point-to-point and reductions |
| `acg-cg-nvshmem` | `cuda_nvshmem` | NVSHMEM halo and reductions |
| `acg-cg-oshmpi` | `cuda_mpi` for reduction only | OSHMPI halo with CUDA-aware `MPI_Allreduce` |
| `acg-cg-oshmpi-backup-mpi` | `cuda_mpi` for reduction only | Archived copy of the same measured operation mapping |

`acg-device-nvshmem` has no direct benchmark equivalent. It uses a different,
device-initiated execution model and does not expose comparable phase timers.

## Reduction prediction

The `cg_step` phase at side 512 is used here. Changing the side to 2048 changes
the stable-backend errors by only a few percentage points because the reduction
payload remains one double.

An OSHMPI row is excluded because the checked-in CG-step data predate the
host-to-device result copy and the measured aCG path uses a different
collective. The experimental implementations require a Leonardo rerun before a
matched prediction error can be reported.

| Backend comparison | Bump MAPE | Queen MAPE | Pearson r | Assessment |
| --- | ---: | ---: | ---: | --- |
| MPI reduction with OSHMPI halo backup | 6.4% | 7.5% | 0.94-0.97 | good |
| NCCL | 10.5% | 7.8% | 0.98-1.00 | good |
| NVSHMEM, configuration matched | 14% | 21% | about 0.96-0.98 | moderate |
| MPI with MPI halo | not reliable | not reliable | not meaningful | halo instability contaminates arrival time |

Using the median aCG trial instead of the fastest one does not change the
stable-backend conclusion: NCCL remains within 7-8%. MPI medians become much
worse because the application's MPI halo has a severe intermittent slow mode.

### MPI

The benchmark predicts CUDA-aware MPI reduction well when it is compared with
the archived aCG configuration that keeps the stable OSHMPI halo: 6.4% MAPE for
Bump and 7.5% for Queen. This isolates the collective from the application's MPI
halo.

The direct `acg-cg-mpi` comparison is not a valid collective test at scale. Its
halo and allreduce event regions inflate together, its trial spread reaches more
than 20x, and selected Bump timing breakdowns at 16 and 32 processes are
self-inconsistent. The benchmark did not predict that failure because its halo
does not reproduce aCG's irregular many-neighbour rendezvous exchange.

### NCCL

NCCL is the strongest validated result. The CG-step reduction predicts aCG to
within 8-11%, with Pearson correlation between 0.98 and 1.00. The exact 8-byte
allreduce microbenchmark gives a similar 10-12% error. NCCL nevertheless performs
better in the full solver than the CG-step headline predicts because aCG can
overlap its asynchronous communication with sparse matrix-vector multiplication.

### NVSHMEM

The checked-in CG-step data and aCG use different collective configurations:
the benchmark permits NCCL dispatch, while aCG sets `NVSHMEM_DISABLE_NCCL=1`.
The unmatched data appear to have only 10-14% error, but that agreement is
accidental. A controlled configuration A/B gives the defensible error of 14%
for Bump and 21% for Queen, and shows that the benchmark remains 40-46%
optimistic at 32 GPUs. It does reproduce the reduction's scale growth: about
5.7x from 2 to 32 GPUs in both the benchmark and aCG.

### OSHMPI

There is no matched OSHMPI scalar-reduction result in the measured campaign.
The checked-in CG-step data call `shmem_double_sum_to_all` on host-staged
operands, whereas every stored aCG OSHMPI result calls CUDA-aware
`MPI_Allreduce`. Comparing their absolute latency would conflate different
collectives. The experimental source changes align both paths around a complete
device-to-host, OpenSHMEM reduction, and host-to-device round trip, but a
Leonardo rerun is required before reporting a prediction error.

## Full-step prediction

For the end-to-end test, side 2048 is the closest available global problem size:
its 4.19 million grid cells are comparable to Bump's 2.91 million and Queen's
4.15 million rows. Within each topology, CG-step and aCG time per iteration are
normalized to NCCL. Correlations are computed over the 15 non-reference
backend/topology ratios.

| Matrix | Pearson r | Mean within-topology rank correlation | Correct fastest backend |
| --- | ---: | ---: | ---: |
| Bump | 0.48 | 0.52 | 2 of 5 topologies |
| Queen | 0.29 | 0.36 | 2 of 5 topologies |

The result is not robust to grid size. CG-step identifies the solver's fastest
backend in 1 of 5 topologies at side 512, 2 of 5 at side 2048, and none at side
8192. At side 2048 it predicts NCCL at 2 and 4 GPUs and OSHMPI at 8-32 GPUs;
aCG measures NCCL as fastest at every scale.

CG-step is therefore not an end-to-end solver-performance model. Its five-point
stencil has four neighbours per row, compared with about 44 nonzeros per row for
Bump and 79 for Queen, and its communication/computation ratio is different.

## Why the halo does not transfer

| Property | CG-step / `halo_1d` | aCG |
| --- | --- | --- |
| neighbours | at most two regular neighbours | up to about 11 irregular neighbours |
| traffic | 4-64 KiB nominal CG-step halo | about 122-328 KiB per process and exchange |
| completion | isolated or blocking | backend-dependent serialization, overlap, or fusion |
| local work | five-point stencil or none | sparse matrix-vector multiplication |

There is no common application halo metric to correlate. aCG's `haloexchange`
timer records only the wait still exposed at `_end`, not total communication.
OSHMPI completes its exchange in `_begin`, so its cost appears in `other`.
NVSHMEM fuses communication into `gemv`. At representative payload sizes,
`halo_1d` generally ranks OSHMPI ahead of NCCL, while aCG consistently ranks
NCCL first.

## Scope statement

> On the same Leonardo software stack, the benchmark predicts scalar-reduction
> performance well for NCCL and CUDA-aware MPI, with approximately 6-11% mean
> error, and moderately for configuration-matched NVSHMEM, with 14-21% error.
> No OSHMPI reduction prediction is reported because the measured benchmark and
> application use different collectives. Neither `cg_step` nor `halo_1d`
> predicts full-solver backend ranking, because the solver uses irregular
> many-neighbour halos with backend-dependent overlap and fusion. The suite is
> therefore validated as a reduction-phase estimator for MPI and NCCL,
> partially for NVSHMEM, but not as an end-to-end solver model.

## Closing the gaps

### 1. Match the staged OSHMPI round trip

The experimental CG-step and aCG worktrees now implement the same sequence:

```text
device partial -> host source -> shmem_double_sum_to_all
              -> host result -> device result
```

Neither change has measured Leonardo results. The next step is to build both
implementations and rerun the matched campaign. If the corrected result still
differs materially, time stream synchronization, device-to-host copy,
collective, and host-to-device copy separately.

### 2. Record and match runtime configuration

Canonical aCG-matched runs need UCC disabled, `NVSHMEM_DISABLE_NCCL=1`, the same
UCX rendezvous threshold and transport settings, and explicit build/runtime
metadata in the result records. Paired variants should run in the same Slurm
allocation when measuring one configuration change.

### 3. Replay aCG's irregular halo

Export each partition's neighbour list and bytes per neighbour from aCG, then
add a trace-driven benchmark that replays that schedule with every backend. It
should measure issue, completion, and exposed wait separately, and optionally
place representative compute between begin and end. This preserves benchmark
control without pretending a two-neighbour ring represents a sparse partition.

### 4. Use a backend-aware step model

A useful compositional model must represent the actual execution modes:

```text
MPI/NCCL:       local work + max(SpMV, halo) + exposed wait + reductions
OSHMPI:         local work + halo + SpMV + reductions
host NVSHMEM:   local work + fused SpMV/communication + reductions
device NVSHMEM: fused device iteration
```

Calibrate any fixed costs on one matrix and predict the other, then reverse the
roles. This prevents a model from claiming accuracy on the data used to fit it.

### 5. Add a sparse mini-application only if needed

If the corrected reduction and trace-driven halo still cannot predict backend
ranking, add a distributed synthetic CSR step with realistic row degree, halo,
dot products, AXPYs, and two scalar reductions. This is the highest-fidelity
option, but it should be last because it approaches duplicating aCG rather than
isolating communication behavior.

Future validation should report per-job medians and confidence intervals beside
best-case performance, and use MAPE, Pearson correlation, rank correlation, and
winner accuracy together. No one statistic captures both calibration and
ordering.
