# Leonardo Halo 1D Experiments

All backends benchmark a **comm-only** 1D halo exchange on Leonardo: a periodic ring where each rank/PE exchanges a halo with both neighbors over a swept halo width, with GPU-resident buffers and slice-local allocation. Every size reports isolated (`batch_iters=1`) and steady-state (`batch_iters=iterations`) results from completed batches.

Args: `<max_halo_elems> <iterations> <warmup> [comma-separated halo sizes]` (e.g. `1048576 100 20`).

Build setup is documented in [`cluster/leonardo/README.md`](../../README.md).

## Topologies

Every backend provides these valid ring topologies. Each launch starts at least
two ranks/PEs.

| Script | Nodes | GPUs/node | Path |
| --- | ---: | ---: | --- |
| `1n2g.sh` | 1 | 2 | intra-node NVLink |
| `1n4g.sh` | 1 | 4 | intra-node NVLink |
| `2n1g.sh` | 2 | 1 | inter-node InfiniBand |
| `2n4g.sh` | 2 | 4 | mixed intra- and inter-node |

## Communication Models

| Backend | Halo Exchange Model |
| --- | --- |
| `cuda_mpi` | Persistent CUDA-aware MPI requests, started and completed once per exchange |
| `cuda_nccl` | Grouped `ncclSend`/`ncclRecv`, with one stream synchronization per batch |
| `cuda_nvshmem` | Persistent cooperative multi-block puts with neighbor completion signals |
| `oshmpi` | GPU-space NBI puts, with `quiet`, CUDA sync, and global barrier per exchange |
| `sycl_mpi` | Persistent SYCL-aware MPI requests, started and completed once per exchange |
| `sycl_oneccl` | Grouped `ccl::send`/`ccl::recv`, with event completion at batch end |

## Submit

Choose a backend directory and one of the four topology scripts. For example:

```bash
GPU_BENCH_N=17 GPU_BENCH_NTRIALS=1 tools/launch.sh halo_1d cuda_mpi 1n2g
GPU_BENCH_N=17 GPU_BENCH_NTRIALS=1 tools/launch.sh halo_1d cuda_mpi 1n4g
GPU_BENCH_N=17 GPU_BENCH_NTRIALS=1 tools/launch.sh halo_1d cuda_mpi 2n1g
GPU_BENCH_N=17 GPU_BENCH_NTRIALS=1 tools/launch.sh halo_1d cuda_mpi 2n4g
```

Backend directories are `cuda_mpi`, `cuda_nccl`, `cuda_nvshmem`, `oshmpi`,
`sycl_mpi`, and `sycl_oneccl`. oneCCL scripts use `mpirun`; all other launchers
use the shared default launcher. `GPU_BENCH_BATCH_SAMPLES` sets the number of
completed batches measured for the `steady` case (default `10`) and
`GPU_BENCH_ISOLATED_SAMPLES` for the `isolated` case (default `100`, since each
of its samples is a single exchange).

Outputs are written to:

```text
results/<result-name>/halo_1d/<job-name>-<job-id>-<trial>-stdout.txt
results/<result-name>/halo_1d/<job-name>-<job-id>-<trial>-stderr.txt
```

## Profiling & Analysis

Set `GPU_BENCH_PROFILE=nsys` to wrap each rank in Nsight Systems and drop one
`.nsys-rep` per rank under `results/<result-name>/halo_1d/profiles/`. Profiling
perturbs timing, so use a dedicated single-trial run and do not report its
numbers:

```bash
GPU_BENCH_PROFILE=nsys GPU_BENCH_NTRIALS=1 tools/launch.sh halo_1d cuda_nvshmem 2n4g
```

`GPU_BENCH_NSYS_TRACE` overrides the trace set (default `cuda,nvtx,mpi`; add `ucx` for
the inter-node IB path). The latency/bandwidth (α–β) model, the
NVSHMEM-vs-NCCL-vs-MPI crossover analysis, and a guide to reading the timelines
are in [`docs/analysis/halo_1d-crossover.md`](../../../../docs/analysis/halo_1d-crossover.md).

## Validated Results

> **Note:** The tables below predate the comm-only ring rewrite of the halo_1d benchmarks (all backends). They reflect the old one-step stencil (single halo width, gathered validation) and are kept only for historical reference. Re-run on Leonardo to regenerate numbers for the new comm-only benchmark.

Validated on Leonardo A100 boost nodes with `GPU_BENCH_N=1048576`. Times are the mean over successful trial stdout files.

### 1 Node / 4 GPUs

| Backend | Ranks/PEs | Trials | Mean Time (s) | Validation |
| --- | ---: | ---: | ---: | --- |
| `cuda_mpi` | 4 ranks | 3 | 0.002873 | PASS |
| `cuda_nccl` | 4 ranks | 3 | 0.023134 | PASS |
| `cuda_nvshmem` | 4 PEs | 3 | 0.000186 | PASS |
| `oshmpi` | 4 PEs | 3 | 0.000763 | PASS |
| `sycl_mpi` | 4 ranks | 3 | 0.002553 | PASS |
| `sycl_oneccl` | 4 ranks | 3 | 0.080008 | PASS |

### 2 Nodes / 8 GPUs

| Backend | Ranks/PEs | Trials | Mean Time (s) | Validation |
| --- | ---: | ---: | ---: | --- |
| `cuda_mpi` | 8 ranks | 3 | 0.014211 | PASS |
| `cuda_nccl` | 8 ranks | 3 | 0.074619 | PASS |
| `cuda_nvshmem` | 8 PEs | 3 | 0.000652 | PASS |
| `oshmpi` | 8 PEs | 3 | 0.002476 | PASS |
| `sycl_mpi` | 8 ranks | 3 | 0.014200 | PASS |
| `sycl_oneccl` | 8 ranks | 3 | 0.063977 | PASS |

### Relative To `cuda_mpi`

Negative delta means faster than `cuda_mpi`; positive delta means slower.

| Backend | 1 Node Delta | 1 Node Speedup | 2 Nodes Delta | 2 Nodes Speedup |
| --- | ---: | ---: | ---: | ---: |
| `cuda_mpi` | 0.0% | 1.00x | 0.0% | 1.00x |
| `cuda_nccl` | +705.2% | 0.12x | +425.1% | 0.19x |
| `cuda_nvshmem` | -93.5% | 15.45x | -95.4% | 21.80x |
| `oshmpi` | -73.4% | 3.77x | -82.6% | 5.74x |
| `sycl_mpi` | -11.1% | 1.13x | -0.1% | 1.00x |
| `sycl_oneccl` | +2684.8% | 0.04x | +350.2% | 0.22x |

Notes:

- SHMEM-related implementations intentionally use one-sided remote writes into ghost cells.
- `sycl_oneccl` is included for completeness but is a collective emulation, not a natural halo exchange.

## Comparability

The implementations exchange and validate the same GPU-resident halos. Backend
completion models remain visible in each report because synchronization costs
differ.

Closest fair comparisons:

- `cuda_mpi` vs `sycl_mpi`: persistent two-sided MPI over device buffers.
- `cuda_nvshmem` vs `oshmpi`: one-sided remote writes, with neighbor signals
  versus a global barrier.
- `cuda_nccl` vs `sycl_oneccl`: queued point-to-point operations with one
  completion boundary per batch.

Important caveats:

- MPI/NCCL/oneCCL are host-submitted, NVSHMEM is device-initiated, and OSHMPI is
  host-initiated one-sided communication.
- OSHMPI includes CUDA device synchronization and a global barrier per exchange
  because CUDA-space RMA completion and passive target-side progress must both
  be closed inside the timed interval.
