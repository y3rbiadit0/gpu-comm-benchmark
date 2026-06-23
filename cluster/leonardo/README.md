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
sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi_cuda_aware_persistent_iter/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/cuda_nvshmem_device/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/sycl_oneccl/1n4g.sh
```

Useful overrides:

```bash
CP_N=16777216 CP_NTRIALS=5 sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n4g.sh
CP_RESULT_NAME=vector-add-sycl-test sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/1n4g.sh
```

Per-problem notes and validated results live in `cluster/leonardo/experiments/<problem>/README.md`.
