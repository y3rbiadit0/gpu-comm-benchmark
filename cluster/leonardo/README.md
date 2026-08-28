# Leonardo

This integration targets Leonardo's NVIDIA A100 `boost_usr_prod` partition. It
provides validated CUDA and SYCL-on-NVIDIA toolchains, dependency bootstraps,
CMake presets, and Slurm runtime settings.

Run all commands from the repository root on a Leonardo login node.

## Prerequisites

The cluster modules provide CUDA, NVHPC, HPC-X, NCCL, NVSHMEM, CMake, and Ninja.
Two SYCL prerequisites must be installed separately:

1. A source build of
   [intel/llvm with NVIDIA CUDA support](https://intel.github.io/llvm/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda).
   Stock oneAPI DPC++ does not include NVPTX support.
2. A local hwloc installation compatible with the DPC++ CUDA adapter.

The default locations are:

```bash
export DPCPP_HOME=$HOME/opt/dpcpp_6.3
export HWLOC_ROOT=$HOME/opt/hwloc
```

The expected compiler path is
`$DPCPP_HOME/llvm/build/install/bin/clang++`. Set `DPCPP_INSTALL` directly when
using another layout. `env/sycl.sh` resolves these compiler and library paths
during environment setup.

## Build

Initialize the dependency stack and build every Leonardo preset with:

```bash
make init
```

This is the complete first-build entry point. It runs the dependency bootstrap
and then builds the CUDA and SYCL preset groups. The individual stages below are
useful when rebuilding only part of the project.

Bootstrap the OSHMPI and oneCCL dependencies. The default bootstrap also builds
the `leonardo-sycl-oneccl-oshmpi` preset:

```bash
make bootstrap
```

The Make target wraps `cluster/leonardo/bootstrap.sh`, which resolves dependency
order, skips installed targets, and supports targeted or forced rebuilds:

```bash
./cluster/leonardo/bootstrap.sh --list
make bootstrap TARGETS=oneccl-nccl
GPU_BENCH_FORCE=1 make bootstrap TARGETS=oneccl-oshmpi
```

Sources and build trees default to `$SCRATCH/gpu-comm-bench`; installed
dependencies default to `$HOME/opt/gpu-comm-bench`. Override
`GPU_BENCH_WORK_ROOT` or `GPU_BENCH_PREFIX_ROOT` before bootstrapping to relocate
them. `layout.sh` is shared by bootstrap, configure, and runtime scripts.

Build all presets or one toolchain group:

```bash
make leonardo
make leonardo-cuda
make leonardo-sycl
```

For a single preset, load its stack and use CMake directly:

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-mpi
cmake --build --preset leonardo-cuda-mpi
```

Use `source cluster/leonardo/environment.sh sycl` for a SYCL preset.

| Preset | Toolchain | Main dependency |
| --- | --- | --- |
| `leonardo-cuda-mpi` | NVHPC 24.5, CUDA 12.4 | HPC-X MPI 2.19 |
| `leonardo-cuda-nccl` | NVHPC 24.5, CUDA 12.4 | NVHPC NCCL |
| `leonardo-cuda-nvshmem` | NVHPC 24.5, CUDA 12.4 | NVHPC NVSHMEM |
| `leonardo-oshmpi` | NVHPC 24.5, CUDA 12.4 | Bootstrapped OSHMPI |
| `leonardo-sycl-mpi` | GCC 12.2, CUDA 12.2, DPC++ | HPC-X MPI 2.19 |
| `leonardo-sycl-oneccl` | GCC 12.2, CUDA 12.2, DPC++ | oneCCL with NCCL |
| `leonardo-sycl-oneccl-oshmpi` | GCC 12.2, CUDA 12.2, DPC++ | oneCCL with OSHMPI |

The SYCL MPI build resolves HPC-X without loading the NVHPC module so that both
MPI implementations use the same MPI bundle without replacing DPC++ or CUDA
12.2. The NCCL-backed oneCCL build uses its bundled MPI and therefore launches
with its matching `mpirun`.

## Run

Set a Slurm account if your Leonardo user has no site default:

```bash
export GPU_BENCH_SLURM_ACCOUNT=<account>
```

Inspect one cell, submit it, or submit a benchmark across its declared matrix:

```bash
cluster/harness/launch.sh --explain allreduce cuda_mpi 1n4g
cluster/harness/launch.sh allreduce cuda_mpi 1n4g
cluster/harness/launch.sh --all allreduce
```

Submit every benchmark with `make submit` or
`cluster/harness/launch.sh --all`. Use `--dry-run --all` before a large
submission. The [harness guide](../harness/README.md) documents filters,
overrides, output paths, and profiling.

## Leonardo Layout

| Path | Responsibility |
| --- | --- |
| `cluster.sh` | Interface used by the shared harness |
| `backends.sh` | Backend, launcher, preset, and binary registry |
| `slurm.sh` | Partition and optional account settings |
| `layout.sh` | Dependency source and installation prefixes |
| `env/` | Build toolchains and modules |
| `runtime/` | UCX, UCC, NCCL, NVSHMEM, MPI, and oneCCL settings |
| `deps/` | Third-party dependency build targets |
| `bootstrap.sh` | Dependency build entry point |

Machine-independent experiment semantics belong under `cluster/harness`, not in
this directory.
