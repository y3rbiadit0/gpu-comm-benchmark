# Communication Playground

This repository collects small distributed GPU communication examples across programming models and communicator libraries.

## Problems

| Problem | Purpose |
| --- | --- |
| `vector_add` | Basic distributed execution |
| `dot_product` | Collective communication |
| `axpy_norm` | Realistic linear algebra building block |
| `halo_1d` | Neighbor communication and one-sided models |
| `cg_step` | Mini conjugate-gradient step connected to solver workloads |

## Implementations

| Implementation | Status |
| --- | --- |
| `cuda_mpi` | `vector_add` implemented |
| `sycl_mpi` | `vector_add` implemented |
| `sycl_oneccl` | Scaffolded |
| `cuda_nccl` | Scaffolded |
| `cuda_nvshmem` | Scaffolded |

## Build

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

## Run

```bash
mpirun -np 4 ./build/cuda-mpi/implementations/cuda_mpi/cuda_mpi_vector_add 1048576
mpirun -np 4 ./build/sycl-mpi/implementations/sycl_mpi/sycl_mpi_vector_add 1048576
```

Cluster-specific setup and experiment scripts live under `cluster/<cluster-name>/`.
