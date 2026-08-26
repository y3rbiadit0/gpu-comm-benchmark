# GPU Communication Benchmark

Comparative benchmarks for GPU communication on HPC clusters: the same workloads
run over several communication models, so the results can be compared directly.

Organized by communication model - message passing (`src/mpi`), one-sided PGAS
(`src/shmem`), and collective libraries (`src/xccl`) - each with CUDA and SYCL
implementations where both apply.

## Layout

```text
src/
  mpi/      # CUDA MPI and SYCL MPI
  xccl/     # CUDA NCCL and SYCL oneCCL
  shmem/    # NVSHMEM and OSHMPI
```

## Benchmarks

The active suite contains six communication patterns, organised in two tiers.

**Microbenchmarks** isolate a single communication operation and sweep message
size, yielding alpha-beta fits (latency floor `alpha`, asymptotic bandwidth
`B_inf`, and `n_half = alpha * B_inf`):

| Benchmark | Purpose |
| --- | --- |
| `pingpong` | Point-to-point one-way latency and bandwidth (message-size sweep) |
| `halo_1d` | Neighbor communication and one-sided models ([guide](docs/halo_1d.md)) |
| `allreduce` | Collective sum latency and bandwidth (message-size sweep) |
| `alltoall` | All-to-all personalized exchange, message-size sweep (per-rank + bus bandwidth) |

**Application benchmarks** fix the problem size and combine several operations
in the order an application issues them. They are single iterations extracted
from real applications, not complete ones. Their axis is rank count, not
message size, so they carry no alpha-beta fit:

| Benchmark | Purpose |
| --- | --- |
| `cg_step` | Conjugate-gradient iteration skeleton (SpMV halo + two reductions) |
| `moe` | Top-1 MoE dispatch + combine with variable, skewed expert traffic |

The second tier exists to test whether the first tier's ranking predicts real
behaviour. Twice so far it has not: Open MPI's UCC collectives are 139x faster
than the default at 16 MiB allreduce yet make `cg_step` 1.90x *slower*, because
that solver's reductions are 8 bytes; and NVSHMEM has the lowest latency floor
in the suite (7.5 us) yet is the slowest backend on `cg_step` at 16 and 32 GPUs.
A suite of microbenchmarks alone would have reported both the wrong way round.

## Implementations

| Model | Backend | Details |
| --- | --- | --- |
| MPI | CUDA | [`src/mpi/cuda`](src/mpi/cuda/README.md) |
| MPI | SYCL | [`src/mpi/sycl`](src/mpi/sycl/README.md) |
| XCCL | CUDA/NCCL | [`src/xccl/cuda`](src/xccl/cuda/README.md) |
| XCCL | SYCL/oneCCL | [`src/xccl/sycl`](src/xccl/sycl/README.md) |
| SHMEM | NVSHMEM | [`src/shmem/nvshmem`](src/shmem/nvshmem/README.md) |
| SHMEM | OSHMPI | [`src/shmem/oshmpi`](src/shmem/oshmpi/README.md) |

Known backend capability gaps, explicit workarounds, and pending validation are
tracked in [`docs/unsupported-operations.md`](docs/unsupported-operations.md).

## Build

Most presets need external libraries that the cluster does not provide. Build those
first with the bootstrap, then build the benchmarks.

### 1. Provide a DPC++ compiler

The one prerequisite the bootstrap cannot install for you. Point `DPCPP_HOME` at an
install (default `$HOME/opt/dpcpp_6.3`); it is needed by every SYCL preset and by the
oneCCL the bootstrap builds. Everything else comes from cluster modules or from the
bootstrap itself.

### 2. Bootstrap the dependencies

```bash
./cluster/leonardo/bootstrap.sh          # or: make bootstrap
```

This clones and builds, in order, patched OSHMPI and oneCCL with the OSHMPI backend,
then the `leonardo-sycl-oneccl-oshmpi` benchmarks. Sources and build trees go to
`$SCRATCH`; install prefixes to `$HOME/opt/gpu-comm-bench`, both defined once in
[`cluster/leonardo/layout.sh`](cluster/leonardo/layout.sh). Re-running skips what is
already installed; `GPU_BENCH_FORCE=1` rebuilds anyway.

```bash
./cluster/leonardo/bootstrap.sh --list           # available targets
./cluster/leonardo/bootstrap.sh oneccl-oshmpi    # one target and its dependencies
```

### 3. Build the benchmarks

```bash
make leonardo          # all Leonardo presets
make leonardo-cuda     # CUDA-stack presets only
make leonardo-sycl     # SYCL-stack presets only
```

Or one preset at a time, which is the fastest loop while editing a benchmark:

```bash
make configure PRESET=cuda-mpi
make build PRESET=cuda-mpi
```

The Makefile is a thin wrapper over `CMakePresets.json`.

### What each preset needs

| Preset | Stack | Depends on | Provided by |
| --- | --- | --- | --- |
| `leonardo-cuda-mpi` | cuda | HPC-X MPI | module |
| `leonardo-cuda-nccl` | cuda | NCCL | `nvhpc` module |
| `leonardo-cuda-nvshmem` | cuda | NVSHMEM | `nvhpc` module |
| `leonardo-oshmpi` | cuda | OSHMPI | `bootstrap.sh oneccl-oshmpi` |
| `leonardo-sycl-mpi` | sycl | Open MPI, DPC++ | module, your DPC++ |
| `leonardo-sycl-oneccl` | sycl | oneCCL with the NCCL backend | `bootstrap.sh oneccl-nccl` -> `$ONECCL_NCCL_ROOT` |
| `leonardo-sycl-oneccl-oshmpi` | sycl | OSHMPI, oneCCL with the OSHMPI backend | `bootstrap.sh` |

A preset whose dependency is missing now fails at configure time naming the path it
searched, rather than at link time with undefined symbols.

## Run Examples

```bash
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_halo_1d 1048576 100 20
mpirun -np 2 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_pingpong 4194304 100 20
mpirun -np 2 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_pingpong 4194304 100 20 1,8,64,1024
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_allreduce 4194304 100 20
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_allreduce 4194304 100 20 1,8,64,1024
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_alltoall 65536 100 20
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_cg_step 512 50 10
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_moe 16384 256 100 20 uniform,locality80,hotspot80
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_halo_1d 1048576 100 20
```

Each binary accepts the global problem size as the first argument. Iterative halo variants also accept an iteration count as the second argument. `pingpong` binaries require exactly 2 ranks, accept `<max_elements> [iterations] [warmup] [message_sizes]`, and sweep message sizes internally (one report line per size). If `[message_sizes]` is omitted, pingpong uses powers of two from 1 to `<max_elements>`; otherwise pass comma-separated element counts such as `1,8,64,1024`. `allreduce` binaries accept `<max_elements> [iterations] [warmup] [message_sizes]` and use the same default power-of-two sweep or optional comma-separated size list. `alltoall` binaries accept `<max_count_per_peer> [iterations] [warmup] [message_sizes]` and sweep per-peer counts the same way `allreduce` does; each rank exchanges `count_per_peer` elements with every rank (send/recv buffers are `ranks × count_per_peer`). Because each rank also sends a block to itself, which never crosses a link, the line carries `bus_gbytes_per_s = algorithm × (P−1)/P` alongside the raw per-rank figure — that correction is zero at one rank, where nothing is communicated. `cg_step` binaries accept `<side> [iterations] [warmup]` and run one CG-iteration communication skeleton per step: an SpMV (column-slab halo exchange + 5-point stencil) followed by two scalar global reductions. `moe` binaries accept `<tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]`; the optional comma-separated routing cases are `uniform`, `locality80`, and `hotspot80`, and omission runs all three internally.

## Benchmark Output Schema

Timed benchmarks built on `include/timing.hpp` and `include/report.hpp` emit a standardized
`key=value` line on the root rank so one parser can compare every backend:

```text
<name> n=<elements> ranks=<n> bytes=<per-iter> iters=<n> warmup=<n> \
  time_per_iter_s=<s> usec=<us> min_usec=<us> max_usec=<us> gbytes_per_s=<gb/s> \
  [median_usec=<us> p25_usec=<us> p75_usec=<us> stddev_usec=<us>] \
  [case=<case>] [status=OK|NOT_IMPLEMENTED|ERROR] validation=PASS|SKIP|FAIL
```

Every rank records the same number of samples. An exchange costs what its
slowest participant took, so the reported series is the element-wise max across
ranks and `time_per_iter_s`/`usec` is the mean of that series. Most benchmarks
record one completed operation per sample. `halo_1d` instead records completed
batches and amortizes each sample by the batch length; this preserves the same
`AVG(MAX per sample)` estimator while supporting queued and persistent work.
`include/stats/collective.hpp` defines the rule; `stats/collective_mpi.hpp` and
`stats/collective_shmem.hpp` carry it out.

`min_usec`, `max_usec` and the optional `median_usec`/`p25_usec`/`p75_usec`/`stddev_usec`
describe that same reduced series, so every field on the line refers to one
distribution. Compare `usec` for small-message latency and `gbytes_per_s` for
bandwidth.

`pingpong` reports **one-way** figures (half the measured round trip): `usec` is one-way
latency and `gbytes_per_s` is one-way bandwidth, with one line per swept message size.
It is the one benchmark not reduced across ranks — the peer's timings cover a
different window (it waits for the ping before replying), so the initiator's
round trip is the measurement by definition.

Each `halo_1d` binary emits `case=isolated` with `batch_iters=1` and
`case=steady` with `batch_iters=<iterations>`. The `steady` case records
`GPU_BENCH_BATCH_SAMPLES` completed-batch samples (default `10`); the `isolated`
case records `GPU_BENCH_ISOLATED_SAMPLES` (default `100`), because each of its
samples is a single exchange and averages nothing internally. Every line carries
its own `batch_samples=`. The two cases are grouped separately by Benchscribe.

MoE emits one line per routing `case`; an unsupported oneCCL point-to-point capability is
reported as `NOT_IMPLEMENTED`/`SKIP` rather than as a timing failure.
`case` is part of the result grouping key. Numeric summaries use only
`status=OK validation=PASS` records; `ERROR` or validation-failed records are
excluded, and contradictory `status=OK validation=FAIL` records are treated as
errors. `NOT_IMPLEMENTED`/`SKIP` records remain visible as `N/I`.

## Summarizing Results

`tools/benchscribe` parses the `results/` tree, aggregates across trials, and
prints a comparison table where every backend is normalized to the `cuda_mpi` baseline:

```bash
python3 tools/benchscribe                       # Markdown, all benchmarks
python3 tools/benchscribe --benchmark allreduce
python3 tools/benchscribe --format csv > summary.csv
```

Figures are drawn from that JSON by [`tools/plot`](tools/plot/README.md), a
separate `uv` project so matplotlib stays out of Benchscribe — message-size
sweeps, bandwidth curves, the α/B∞/n½ bars, and a speedup heatmap:

```bash
python3 tools/benchscribe --format json       > points.json
python3 tools/benchscribe --fit --format json > fit.json
uv run --project tools/plot gpu-bench-plot --points points.json --fit fit.json \
    --benchmark halo_1d --outdir docs/analysis/figures
```

See [`tools/README.md`](tools/README.md) for details.

## Leonardo

Use the Slurm scripts for validation runs; they request the GPU partition and resources correctly.

```bash
GPU_BENCH_N=17 GPU_BENCH_NTRIALS=1 cluster/harness/launch.sh halo_1d cuda_mpi 1n2g
GPU_BENCH_MSG_SIZES=1,1024 GPU_BENCH_NTRIALS=1 cluster/harness/launch.sh allreduce cuda_mpi 1n4g
GPU_BENCH_ROUTINGS=uniform,hotspot80 GPU_BENCH_NTRIALS=1 cluster/harness/launch.sh moe cuda_mpi 1n4g
```

Leonardo setup, presets, and experiment notes live under [`cluster/leonardo`](cluster/leonardo/README.md).
