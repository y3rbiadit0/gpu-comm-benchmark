# Leonardo Halo 1D Experiments

All backends benchmark a **comm-only** 1D halo exchange on Leonardo: a periodic ring where each rank/PE exchanges a halo with both neighbors over a swept halo width, with GPU-resident buffers and slice-local allocation. Timing uses the shared warmup/iteration harness and reports the slowest-rank average per halo width (`print_report` format).

Args: `<max_halo_elems> <iterations> <warmup> [comma-separated halo sizes]` (e.g. `1048576 100 20`).

Build setup is documented in [`cluster/leonardo/README.md`](../../README.md).

## Topologies

Every backend, including `cuda_nvshmem_optimized`, provides these valid ring
topologies. Each launch starts at least two ranks/PEs.

| Script | Nodes | GPUs/node | Path |
| --- | ---: | ---: | --- |
| `1n2g.sh` | 1 | 2 | intra-node NVLink |
| `1n4g.sh` | 1 | 4 | intra-node NVLink |
| `2n1g.sh` | 2 | 1 | inter-node InfiniBand |
| `2n4g.sh` | 2 | 4 | mixed intra- and inter-node |

## Communication Models

| Backend | Halo Exchange Model |
| --- | --- |
| `cuda_mpi` | CUDA-aware `MPI_Isend`/`MPI_Irecv`/`MPI_Waitall` neighbor exchange (comm-only ring) |
| `cuda_nccl` | Grouped `ncclSend`/`ncclRecv` with both neighbors (comm-only ring) |
| `cuda_nvshmem` | Device-initiated `nvshmemx_float_put_signal_nbi_block` + `nvshmem_signal_wait_until` P2P sync (comm-only ring) |
| `oshmpi` | One-sided `shmem_putmem` + `shmem_quiet` + global barrier completion (comm-only ring) |
| `sycl_mpi` | SYCL-aware `MPI_Isend`/`MPI_Irecv`/`MPI_Waitall` neighbor exchange (comm-only ring) |
| `sycl_oneccl` | Grouped point-to-point `ccl::send`/`ccl::recv` with both neighbors through native NCCL groups |

## Submit

Choose a backend directory and one of the four topology scripts. For example:

```bash
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/1n2g.sh
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/1n4g.sh
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/2n1g.sh
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/2n4g.sh
```

Backend directories are `cuda_mpi`, `cuda_nccl`, `cuda_nvshmem`,
`cuda_nvshmem_optimized`, `oshmpi`, `sycl_mpi`, and `sycl_oneccl`. oneCCL
scripts use `mpirun`; all other launchers use the shared default launcher.

Outputs are written to:

```text
results/<result-name>/halo_1d/<job-name>-<job-id>-<trial>-stdout.txt
results/<result-name>/halo_1d/<job-name>-<job-id>-<trial>-stderr.txt
```

## Profiling & Analysis

Set `CP_PROFILE=nsys` to wrap each rank in Nsight Systems and drop one
`.nsys-rep` per rank under `results/<result-name>/halo_1d/profiles/`. Profiling
perturbs timing, so use a dedicated single-trial run and do not report its
numbers:

```bash
CP_PROFILE=nsys CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_nvshmem/2n4g.sh
```

`CP_NSYS_TRACE` overrides the trace set (default `cuda,nvtx,mpi`; add `ucx` for
the inter-node IB path). The latency/bandwidth (α–β) model, the
NVSHMEM-vs-NCCL-vs-MPI crossover analysis, and a guide to reading the timelines
are in [`docs/analysis/halo_1d-crossover.md`](../../../../docs/analysis/halo_1d-crossover.md).

## Validated Results

> **Note:** The tables below predate the comm-only ring rewrite of the halo_1d benchmarks (all backends). They reflect the old one-step stencil (single halo width, gathered validation) and are kept only for historical reference. Re-run on Leonardo to regenerate numbers for the new comm-only benchmark.

Validated on Leonardo A100 boost nodes with `CP_N=1048576`. Times are the mean over successful trial stdout files.

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

The implementations solve the same numerical stencil and validate the same output, but they are not all strict apples-to-apples communication benchmarks.

Closest fair comparisons:

- `cuda_mpi` vs `sycl_mpi`: both use host-side `MPI_Sendrecv` for boundary exchange, then accelerator stencil compute.
- `cuda_nvshmem` vs `oshmpi`: both use SHMEM-style one-sided remote writes into neighbor ghost cells and report max elapsed across PEs.
- `cuda_nccl` stands as the native NCCL point-to-point version: GPU-resident `ncclSend`/`ncclRecv` boundary exchange plus NCCL point-to-point gather.

Important caveats:

- The MPI variants are portable host-mediated baselines, not CUDA-aware/SYCL-aware device-buffer MPI halo implementations.
- `sycl_oneccl` uses full-buffer collective emulation with `allreduce(sum)`, so it is useful for correctness and model coverage but should not be treated as a natural halo-exchange performance competitor.
