# Leonardo Halo 1D Experiments

These jobs benchmark a one-step 1D halo stencil on Leonardo. Each rank or PE owns a contiguous interior segment, exchanges one boundary value with each neighbor, computes the stencil on an accelerator, and validates the gathered result on rank/PE 0.

## Communication Models

| Backend | Halo Exchange Model |
| --- | --- |
| `cuda_mpi` | MPI two-sided neighbor exchange with `MPI_Sendrecv` |
| `sycl_mpi` | MPI two-sided neighbor exchange with `MPI_Sendrecv` |
| `cuda_nccl` | NCCL point-to-point `ncclSend`/`ncclRecv` with neighbors |
| `cuda_nvshmem` | One-sided NVSHMEM puts into neighbor ghost cells |
| `oshmpi` | One-sided OSHMPI `shmem_putmem` into neighbor ghost cells |
| `sycl_oneccl` | Collective emulation with `allreduce(sum)` because oneCCL/NCCL has no natural neighbor halo primitive |

## Submit

```bash
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/1n4g.sh
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_nvshmem/1n4g.sh
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/oshmpi/1n4g.sh
```

```bash
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/sycl_mpi/1n4g.sh
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/sycl_oneccl/1n4g.sh
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_nccl/1n4g.sh
```

Outputs are written to:

```text
results/<result-name>/halo_1d/<job-name>-<job-id>-<trial>-stdout.txt
results/<result-name>/halo_1d/<job-name>-<job-id>-<trial>-stderr.txt
```

## Validated Results

Validated on Leonardo A100 boost nodes with `CP_N=1048576`. Times are the mean over successful trial stdout files.

### 1 Node / 4 GPUs

| Backend | Ranks/PEs | Trials | Mean Time (s) | Validation |
| --- | ---: | ---: | ---: | --- |
| `cuda_mpi` | 4 ranks | 3 | 0.002838 | PASS |
| `cuda_nvshmem` | 4 PEs | 3 | 0.000187 | PASS |
| `oshmpi` | 4 PEs | 3 | 0.000770 | PASS |

### 2 Nodes / 8 GPUs

| Backend | Ranks/PEs | Trials | Mean Time (s) | Validation |
| --- | ---: | ---: | ---: | --- |
| `cuda_mpi` | 8 ranks | 3 | 0.013222 | PASS |
| `cuda_nvshmem` | 8 PEs | 3 | 0.000426 | PASS |
| `oshmpi` | 8 PEs | 3 | 0.002091 | PASS |

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
