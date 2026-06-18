# Leonardo

Leonardo uses different validated software stacks for native CUDA and SYCL-on-NVIDIA runs. Select the stack explicitly before configuring or running.

## CUDA Stack

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-mpi
cmake --build --preset leonardo-cuda-mpi
```

This stack is based on `nvhpc/24.5`, `hpcx-mpi/2.19`, CUDA 12.4 from NVHPC, and NVHPC-provided NCCL/NVSHMEM.

## SYCL Stack

```bash
source cluster/leonardo/environment.sh sycl
cmake --preset leonardo-sycl-mpi
cmake --build --preset leonardo-sycl-mpi
```

This stack is based on `gcc/12.2.0`, `cuda/12.2`, `openmpi/4.1.6`, and a custom DPC++ install at `$HOME/opt/dpcpp_6.3`.

## Vector Add Experiments

Submit fixed Leonardo topologies:

```bash
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/2n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/cuda_nvshmem/1n4g.sh
```

```bash
sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/1n1g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/2n4g.sh
sbatch cluster/leonardo/experiments/vector_add/sycl_oneccl/1n4g.sh
```

Useful overrides:

```bash
CP_N=16777216 CP_NTRIALS=5 sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n4g.sh
CP_RESULT_NAME=vector-add-sycl-test sbatch cluster/leonardo/experiments/vector_add/sycl_mpi/1n4g.sh
```

Results are written under:

```text
results/<result-name>/vector_add/
```

Interactive wrappers are still available:

```bash
NP=4 N=1048576 cluster/leonardo/experiments/vector_add/run_cuda_mpi.sh
NP=4 N=1048576 cluster/leonardo/experiments/vector_add/run_sycl_mpi.sh
```

Under Slurm, the wrappers use `srun`; otherwise they fall back to `mpirun`.
