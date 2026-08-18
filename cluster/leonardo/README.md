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
The active suite is `pingpong`, `halo_1d`, `allreduce`, `alltoall`, `cg_step`, and `moe`.

Halo 1D:

```bash
sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_1d/sycl_oneccl/1n4g.sh
```

Allreduce (collective sum latency/bandwidth, internal size sweep):

```bash
sbatch cluster/leonardo/experiments/allreduce/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/allreduce/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/allreduce/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/allreduce/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/allreduce/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/allreduce/sycl_oneccl/1n4g.sh
```

Each `allreduce` backend also has `1n1g`, `1n2g` (NVLink), `2n1g` (InfiniBand), and `2n4g`
launchers; see [`experiments/allreduce/README.md`](experiments/allreduce/README.md).

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

MoE (top-1 variable-count dispatch + combine under uniform, local, and hotspot routing):

```bash
sbatch cluster/leonardo/experiments/moe/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/moe/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/moe/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/moe/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/moe/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/moe/sycl_oneccl/1n4g.sh
```

MoE is a skew-sensitive global personalized application pattern: unlike dense `alltoall`,
its per-peer operation sizes and expert receive loads vary with routing. Each backend has
`1n1g`, `1n2g`, `1n4g`, `2n1g`, and `2n4g` launchers; see
[`experiments/moe/README.md`](experiments/moe/README.md). oneCCL launchers use `mpirun`.

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
GPU_BENCH_N=16777216 GPU_BENCH_NTRIALS=5 sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/1n4g.sh
GPU_BENCH_RESULT_NAME=halo-sycl-test sbatch cluster/leonardo/experiments/halo_1d/sycl_mpi/1n4g.sh
GPU_BENCH_ITERS=500 GPU_BENCH_WARMUP=100 sbatch cluster/leonardo/experiments/allreduce/cuda_nccl/2n1g.sh
GPU_BENCH_HIDDEN=512 GPU_BENCH_ROUTINGS=uniform,hotspot80 sbatch cluster/leonardo/experiments/moe/cuda_nccl/2n4g.sh
```

Per-problem notes and validated results live in `cluster/leonardo/experiments/<problem>/README.md`.
