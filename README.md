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

The active suite contains six communication patterns:

| Benchmark | Purpose |
| --- | --- |
| `pingpong` | Point-to-point one-way latency and bandwidth (message-size sweep) |
| `halo_1d` | Neighbor communication and one-sided models ([guide](docs/halo_1d.md)) |
| `allreduce` | Collective sum latency and bandwidth (message-size sweep) |
| `alltoall` | All-to-all personalized exchange (bisection bandwidth) |
| `cg_step` | Conjugate-gradient iteration skeleton (SpMV halo + two reductions) |
| `moe` | Top-1 MoE dispatch + combine with variable, skewed expert traffic |

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
| `leonardo-sycl-oneccl` | sycl | oneCCL with the NCCL backend | **no bootstrap target yet** - hand-built at `$ONECCL_NCCL_ROOT` |
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

Each binary accepts the global problem size as the first argument. Iterative halo variants also accept an iteration count as the second argument. `pingpong` binaries require exactly 2 ranks, accept `<max_elements> [iterations] [warmup] [message_sizes]`, and sweep message sizes internally (one report line per size). If `[message_sizes]` is omitted, pingpong uses powers of two from 1 to `<max_elements>`; otherwise pass comma-separated element counts such as `1,8,64,1024`. `allreduce` binaries accept `<max_elements> [iterations] [warmup] [message_sizes]` and use the same default power-of-two sweep or optional comma-separated size list. `alltoall` binaries accept `<count_per_peer> [iterations] [warmup]`; each rank exchanges `count_per_peer` elements with every rank (send/recv buffers are `ranks × count_per_peer`). `cg_step` binaries accept `<side> [iterations] [warmup]` and run one CG-iteration communication skeleton per step: an SpMV (column-slab halo exchange + 5-point stencil) followed by two scalar global reductions. `moe` binaries accept `<tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]`; the optional comma-separated routing cases are `uniform`, `locality80`, and `hotspot80`, and omission runs all three internally.

## Benchmark Output Schema

Timed benchmarks built on `include/timing.hpp` and `include/report.hpp` emit a standardized
`key=value` line on the root rank so one parser can compare every backend:

```text
<name> n=<elements> ranks=<n> bytes=<per-iter> iters=<n> warmup=<n> \
  time_per_iter_s=<s> usec=<us> min_usec=<us> max_usec=<us> gbytes_per_s=<gb/s> \
  [case=<case>] [status=OK|NOT_IMPLEMENTED|ERROR] validation=PASS|SKIP|FAIL
```

`time_per_iter_s`/`usec` is the slowest-rank average over the timed loop (after warmup);
`min_usec`/`max_usec` bound the per-iteration distribution. Compare `usec` for
small-message latency and `gbytes_per_s` for bandwidth.

`pingpong` reports **one-way** figures (half the measured round trip): `usec` is one-way
latency and `gbytes_per_s` is one-way bandwidth, with one line per swept message size.
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

See [`tools/README.md`](tools/README.md) for details.

## Leonardo

Use the Slurm scripts for validation runs; they request the GPU partition and resources correctly.

```bash
GPU_BENCH_N=17 GPU_BENCH_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/1n2g.sh
GPU_BENCH_MSG_SIZES=1,1024 GPU_BENCH_NTRIALS=1 sbatch cluster/leonardo/experiments/allreduce/cuda_mpi/1n4g.sh
GPU_BENCH_ROUTINGS=uniform,hotspot80 GPU_BENCH_NTRIALS=1 sbatch cluster/leonardo/experiments/moe/cuda_mpi/1n4g.sh
```

Leonardo setup, presets, and experiment notes live under [`cluster/leonardo`](cluster/leonardo/README.md).
