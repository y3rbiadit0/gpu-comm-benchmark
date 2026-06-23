# Communication Playground

Small distributed GPU communication examples organized by communication model.

## Layout

```text
src/
  mpi/      # CUDA MPI and SYCL MPI
  xccl/     # CUDA NCCL and SYCL oneCCL
  shmem/    # NVSHMEM and OSHMPI
```

## Benchmarks

| Benchmark | Purpose |
| --- | --- |
| `vector_add` | Basic distributed execution |
| `halo_1d` | Neighbor communication and one-sided models |

## Implementations

| Model | Backend | Details |
| --- | --- |
| MPI | CUDA | [`src/mpi/cuda`](src/mpi/cuda/README.md) |
| MPI | SYCL | [`src/mpi/sycl`](src/mpi/sycl/README.md) |
| XCCL | CUDA/NCCL | [`src/xccl/cuda`](src/xccl/cuda/README.md) |
| XCCL | SYCL/oneCCL | [`src/xccl/sycl`](src/xccl/sycl/README.md) |
| SHMEM | NVSHMEM | [`src/shmem/nvshmem`](src/shmem/nvshmem/README.md) |
| SHMEM | OSHMPI | [`src/shmem/oshmpi`](src/shmem/oshmpi/README.md) |

## Build

Build one preset directly:

```bash
make configure PRESET=cuda-mpi
make build PRESET=cuda-mpi
```

Build all Leonardo presets on Leonardo:

```bash
make leonardo
```

Use `make leonardo-cuda` or `make leonardo-sycl` to build only one stack. The Makefile is only a thin wrapper over `CMakePresets.json`.

## Run Examples

```bash
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_vector_add 1048576
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_halo_1d 1048576
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_vector_add 1048576
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_halo_1d 1048576
```

Each binary accepts the global problem size as the first argument. Iterative halo variants also accept an iteration count as the second argument.

## Leonardo

Use the Slurm scripts for validation runs; they request the GPU partition and resources correctly.

```bash
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n1g.sh
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/1n1g.sh
```

Leonardo setup, presets, and experiment notes live under [`cluster/leonardo`](cluster/leonardo/README.md).
