# Leonardo Vector Add Experiments

These jobs follow the same style as the Leonardo experiment scripts in `acg-sycl`: fixed topology launchers, Slurm logs under `logs/`, and per-trial benchmark output under `results/`.

## Build

CUDA + MPI:

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-mpi
cmake --build --preset leonardo-cuda-mpi
```

SYCL + MPI:

```bash
source cluster/leonardo/environment.sh sycl
cmake --preset leonardo-sycl-mpi
cmake --build --preset leonardo-sycl-mpi
```

## Submit

```bash
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/2n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_nccl/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_nccl/2n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_nvshmem/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_nvshmem/2n4g.sh
sbatch cluster/leonardo/experiments/vector_add/oshmpi/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/oshmpi/2n4g.sh
```

```bash
sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/2n4g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_oneccl/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_oneccl/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_oneccl/2n4g.sh
```

## Overrides

```bash
CP_N=16777216
CP_NTRIALS=5
CP_RESULT_NAME=vector-add-cuda-mpi-custom
CP_BINARY=/path/to/vector_add_binary
```

Example:

```bash
CP_N=16777216 CP_NTRIALS=5 sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n4g.sh
```

Outputs are written to:

```text
results/<result-name>/vector_add/<job-name>-<job-id>-<trial>-stdout.txt
results/<result-name>/vector_add/<job-name>-<job-id>-<trial>-stderr.txt
```

## Validated Results

Validated on Leonardo A100 boost nodes with `CP_N=1048576`. Times are the mean over available successful trial stdout files.

### 1 Node / 4 GPUs

| Backend | Ranks/PEs | Trials | Mean Time (s) | Validation |
| --- | ---: | ---: | ---: | --- |
| `cuda_mpi` | 4 ranks | 3 | 0.007282 | PASS |
| `cuda_nccl` | 4 ranks | 3 | 0.020193 | PASS |
| `cuda_nvshmem` | 4 PEs | 3 | 0.000632 | PASS |
| `oshmpi` | 4 PEs | 1 | 0.001831 | PASS |
| `sycl_mpi` | 4 ranks | 3 | 0.003606 | PASS |
| `sycl_oneccl` | 4 ranks | 3 | 0.082637 | PASS |

### 2 Nodes / 8 GPUs

| Backend | Ranks/PEs | Trials | Mean Time (s) | Validation |
| --- | ---: | ---: | ---: | --- |
| `cuda_mpi` | 8 ranks | 3 | 0.019709 | PASS |
| `cuda_nccl` | 8 ranks | 3 | 0.079309 | PASS |
| `cuda_nvshmem` | 8 PEs | 3 | 0.003861 | PASS |
| `oshmpi` | 8 PEs | 3 | 0.004353 | PASS |
| `sycl_mpi` | 8 ranks | 3 | 0.024260 | PASS |
| `sycl_oneccl` | 8 ranks | 3 | 0.062249 | PASS |

Notes:

- `sycl_oneccl` uses the UNISA NCCL-enabled oneCCL fork with bundled Intel MPI on Leonardo.
- The oneCCL NCCL backend does not implement `broadcast`, so `sycl_oneccl` emulates rank-0 input distribution with `allreduce(sum)` over zero-filled non-root buffers.
- Multi-node `sycl_oneccl` requires bundled Intel MPI OFI/TCP startup settings from `cluster/leonardo/runtime/oneccl-nccl.sh`.
- `cuda_nvshmem` and `oshmpi` use comparable result collection/timing semantics: local slices are gathered to PE 0, and timings report max elapsed across PEs.
