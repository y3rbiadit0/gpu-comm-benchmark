# Leonardo

Leonardo uses different validated software stacks for native CUDA and SYCL-on-NVIDIA runs. Select the stack explicitly before configuring or submitting jobs.

## CUDA Stack

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-mpi
cmake --build --preset leonardo-cuda-mpi
```

This stack is based on `nvhpc/24.5`, `hpcx-mpi/2.19`, CUDA 12.4 from NVHPC, and NVHPC-provided NCCL/NVSHMEM.

Other CUDA presets:

```bash
cmake --preset leonardo-cuda-nccl && cmake --build --preset leonardo-cuda-nccl
cmake --preset leonardo-cuda-nvshmem && cmake --build --preset leonardo-cuda-nvshmem
cmake --preset leonardo-oshmpi && cmake --build --preset leonardo-oshmpi
```

## SYCL Stack

```bash
source cluster/leonardo/environment.sh sycl
cmake --preset leonardo-sycl-mpi
cmake --build --preset leonardo-sycl-mpi
```

This stack is based on `gcc/12.2.0`, `cuda/12.2`, `openmpi/4.1.6`, and a custom DPC++ install at `$HOME/opt/dpcpp_6.3`.

oneCCL preset:

```bash
cmake --preset leonardo-sycl-oneccl
cmake --build --preset leonardo-sycl-oneccl
```

## Experiments

Experiment scripts are fixed-topology Slurm launchers. They write Slurm logs under `logs/` and benchmark outputs under `results/<result-name>/<problem>/`.

Vector add:

```bash
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/2n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_oneccl/1n4g.sh
```

Halo 1D:

```bash
sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi_cuda_aware_iter/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/cuda_nvshmem_device/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/sycl_oneccl/1n4g.sh
```

Dot product (global reduction / allreduce latency):

```bash
sbatch cluster/leonardo/experiments/dot_product/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/dot_product/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/dot_product/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/dot_product/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/dot_product/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/dot_product/sycl_oneccl/1n4g.sh
```

Each `dot_product` backend also has `1n1g`, `1n2g` (NVLink), `2n1g` (InfiniBand), and `2n4g`
launchers; see [`experiments/dot_product/README.md`](experiments/dot_product/README.md).

Ping-pong (point-to-point one-way latency/bandwidth, 2 endpoints, internal size sweep):

```bash
sbatch cluster/leonardo/experiments/pingpong/cuda_mpi/1n2g.sh   # NVLink
sbatch cluster/leonardo/experiments/pingpong/cuda_mpi/2n1g.sh   # InfiniBand
sbatch cluster/leonardo/experiments/pingpong/cuda_nccl/2n1g.sh
sbatch cluster/leonardo/experiments/pingpong/cuda_nvshmem/2n1g.sh
sbatch cluster/leonardo/experiments/pingpong/oshmpi/2n1g.sh
sbatch cluster/leonardo/experiments/pingpong/sycl_mpi/2n1g.sh
sbatch cluster/leonardo/experiments/pingpong/sycl_oneccl/2n1g.sh
```

Only the `1n2g` (intra-node NVLink) and `2n1g` (inter-node InfiniBand) topologies exist for
`pingpong`; see [`experiments/pingpong/README.md`](experiments/pingpong/README.md).

Halo 2D (5-point Jacobi stencil, strided column halo exchange):

```bash
sbatch cluster/leonardo/experiments/halo_2d/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_2d/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/halo_2d/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/halo_2d/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_2d/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_2d/sycl_oneccl/1n4g.sh
```

Each `halo_2d` backend has `1n1g`, `1n2g`, `1n4g`, `2n1g`, and `2n4g` launchers; see
[`experiments/halo_2d/README.md`](experiments/halo_2d/README.md).

All-to-all (personalized exchange / bisection bandwidth):

```bash
sbatch cluster/leonardo/experiments/alltoall/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/alltoall/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/alltoall/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/alltoall/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/alltoall/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/alltoall/sycl_oneccl/1n4g.sh
```

Each `alltoall` backend has `1n1g`, `1n2g`, `1n4g`, `2n1g`, and `2n4g` launchers; see
[`experiments/alltoall/README.md`](experiments/alltoall/README.md).

CG step (conjugate-gradient iteration skeleton: SpMV halo + two reductions):

```bash
sbatch cluster/leonardo/experiments/cg_step/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/cg_step/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/cg_step/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/cg_step/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/cg_step/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/cg_step/sycl_oneccl/1n4g.sh
```

Each `cg_step` backend has `1n1g`, `1n2g`, `1n4g`, `2n1g`, and `2n4g` launchers; see
[`experiments/cg_step/README.md`](experiments/cg_step/README.md).

Useful overrides:

```bash
CP_N=16777216 CP_NTRIALS=5 sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n4g.sh
CP_RESULT_NAME=vector-add-sycl-test sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/1n4g.sh
CP_ITERS=500 CP_WARMUP=100 sbatch cluster/leonardo/experiments/dot_product/cuda_nccl/2n1g.sh
```

Per-problem notes and validated results live in `cluster/leonardo/experiments/<problem>/README.md`.
