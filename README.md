# Communication Playground

Small distributed GPU communication examples across programming models and communication libraries.

## Problems

| Problem | Purpose |
| --- | --- |
| `vector_add` | Basic distributed execution |
| `halo_1d` | Neighbor communication and one-sided models |

Planned problems are documented when they get an implementation.

## Implementations

| Implementation | Targets | Details |
| --- | --- |
| `cuda_mpi` | `vector_add`, `halo_1d`, iterative CUDA-aware halo variants | [`src/cuda_mpi`](src/cuda_mpi/README.md) |
| `sycl_mpi` | `vector_add`, `halo_1d` | [`src/sycl_mpi`](src/sycl_mpi/README.md) |
| `cuda_nccl` | `vector_add`, `halo_1d` | [`src/cuda_nccl`](src/cuda_nccl/README.md) |
| `cuda_nvshmem` | `vector_add`, `halo_1d`, host/device communication variants | [`src/cuda_nvshmem`](src/cuda_nvshmem/README.md) |
| `sycl_oneccl` | `vector_add`, `halo_1d` collective emulation | [`src/sycl_oneccl`](src/sycl_oneccl/README.md) |
| `oshmpi` | `vector_add`, `halo_1d` | [`src/oshmpi`](src/oshmpi/README.md) |

## Local Build

CUDA + MPI:

```bash
cmake --preset cuda-mpi
cmake --build --preset cuda-mpi
```

SYCL + MPI:

```bash
cmake --preset sycl-mpi -DCMAKE_CXX_COMPILER=icpx
cmake --build --preset sycl-mpi
```

Optional backends have additional dependencies and presets. See `CMakePresets.json` for the full preset list.

## Local Run

```bash
mpirun -np 4 ./build/cuda-mpi/src/cuda_mpi/cuda_mpi_vector_add 1048576
mpirun -np 4 ./build/cuda-mpi/src/cuda_mpi/cuda_mpi_halo_1d 1048576

mpirun -np 4 ./build/sycl-mpi/src/sycl_mpi/sycl_mpi_vector_add 1048576
mpirun -np 4 ./build/sycl-mpi/src/sycl_mpi/sycl_mpi_halo_1d 1048576
```

Each binary accepts the global problem size as the first argument. Iterative halo variants also accept an iteration count as the second argument.

## Leonardo

Leonardo setup and Slurm experiment launchers live under [`cluster/leonardo`](cluster/leonardo/README.md).
