# Leonardo

This integration targets Leonardo's NVIDIA A100 `boost_usr_prod` partition. It
provides validated CUDA and SYCL-on-NVIDIA toolchains, dependency bootstraps,
CMake presets, and Slurm runtime settings.

Run all commands from the repository root on a Leonardo login node.

## 📋 Prerequisites

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

## 🔧 Build

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

## 📡 Communication libraries

Find your backend, read the last column, follow that recipe.

| Backend | Library | Swap it? | How |
| --- | --- | --- | --- |
| `cuda_nvshmem` | NVSHMEM 2.11, or any release | ✅ per run | [Recipe A](#-recipe-a-swap-a-library-version) |
| `cuda_nccl` | NCCL (NVHPC 24.5), or any release | ✅ per run | [Recipe A](#-recipe-a-swap-a-library-version) |
| `oshmpi`, `sycl_oneccl_oshmpi` | OSHMPI at a pinned ref | ⚠️ rebuild | [Recipe B](#-recipe-b-swap-a-pinned-git-ref) |
| `sycl_oneccl` | oneCCL at a pinned ref, on the `nccl` module | ⚠️ rebuild | [Recipe B](#-recipe-b-swap-a-pinned-git-ref) |
| `cuda_mpi`, `sycl_mpi` | HPC-X MPI 2.19 | 🔒 module | [Recipe C](#-recipe-c-swap-a-module-provided-library) |

✅ means two choices coexist: separate prefixes, separate build trees, separate
results trees. ⚠️ means they share all three -- you must rebuild and keep the
results apart yourself. 🔒 means the version belongs to the site's module.

### ✅ Recipe A: swap a library version

Two libraries are wired this way, NVSHMEM and NCCL. Both work the same: set one
variable, then run the four steps in order. Everything downstream follows from
it -- prefix, build tree, results tree, and the transport default.

#### NVSHMEM

```bash
export GPU_BENCH_NVSHMEM_VERSION=3.7.2                # 1. name the release

./cluster/leonardo/bootstrap.sh nvshmem               # 2. fetch and verify it
make leonardo-cuda                                    # 3. build against it
cluster/harness/launch.sh --all halo_1d               # 4. measure it

python3 tools/benchscribe results-nvshmem-3.7.2 --benchmark halo_1d
```

Unset the variable to go back to the module's 2.11. Do not clean anything in
between -- the two selections never touch the same path:

| | default (`module`) | `GPU_BENCH_NVSHMEM_VERSION=3.7.2` |
| --- | --- | --- |
| 📦 library | `nvhpc` module's NVSHMEM 2.11 | `$HOME/opt/gpu-comm-bench/nvshmem-3.7.2` |
| 🔨 build | `build/leonardo-cuda-nvshmem` | `build-nvshmem-3.7.2/leonardo-cuda-nvshmem` |
| 📈 results | `results/` | `results-nvshmem-3.7.2/` |

Check the resolution before you submit anything:

```bash
cluster/harness/launch.sh --explain halo_1d cuda_nvshmem 2n4g
```

Four things to know:

- Both runs report themselves as the `cuda_nvshmem` backend. They are the same
  backend; the version is a property of the run, recorded in the path and in
  each log's `ENV` block (`GPU_BENCH_NVSHMEM_VERSION`, `NVSHMEM_HOME`).
- Step 2 downloads ~300 MB, verifies the checksum against NVIDIA's published
  manifest, and probes that the stack's `nvcc` can device-link the result before
  declaring success. Pass `GPU_BENCH_NVSHMEM_SHA256` for a release with no
  checksum in `deps/nvshmem.sh`.
- A 3.x selection switches the transport default to CPU-assisted IBGDA, which is
  the reason to run one here at all. Set `NVSHMEM_REMOTE_TRANSPORT=ibrc` to hold
  the transport constant and compare versions on the same proxy path.
- Reach for this whenever the question is "does this library version change the
  answer".

#### NCCL

Same four steps and the same guarantees, with a source build in place of a
redistributable:

```bash
export GPU_BENCH_NCCL_VERSION=2.31.2-1                # 1. name the release

./cluster/leonardo/bootstrap.sh nccl                  # 2. clone, build, install
make leonardo-cuda                                    # 3. build against it
cluster/harness/launch.sh --all allreduce             # 4. measure it

python3 tools/benchscribe results-nccl-2.31.2-1 --benchmark allreduce
```

| | default (`module`) | `GPU_BENCH_NCCL_VERSION=2.31.2-1` |
| --- | --- | --- |
| 📦 library | `nvhpc` module's NCCL | `$HOME/opt/gpu-comm-bench/nccl-2.31.2-1` |
| 🔨 build | `build/leonardo-cuda-nccl` | `build-nccl-2.31.2-1/leonardo-cuda-nccl` |
| 📈 results | `results/` | `results-nccl-2.31.2-1/` |

Four things to know:

- The version string is NCCL's own release name, package revision included
  (`2.31.2-1`), because the git tag it clones is `v<version>`. Point
  `GPU_BENCH_NCCL_REF` at a branch or commit to build something that is not a
  release.
- Step 2 builds against the toolkit `env/cuda.sh` selected, so the library and
  the benchmarks that call it agree on CUDA. It compiles for `sm_80` only, tries
  the stack's `nvc++` as the host compiler and falls back to `g++` if NCCL's
  build rejects it (`GPU_BENCH_NCCL_CXX` pins either), and finishes by compiling
  NCCL's own GIN example with this stack's `nvcc` -- if that fails, the device
  API is not usable here and the target says so instead of leaving it to be
  discovered in a preset build.
- The reason to run one at all is the device API (`nccl_device.h`): since 2.28,
  kernels can initiate communication themselves, over NVLink (LSA) and over the
  NIC (GIN). The module's NCCL predates it. Calling it needs new benchmark
  sources -- installing the library does not change what the current ones do,
  which measure the host API and will simply run against a newer NCCL.
- This is a cuda-stack selection. `sycl_oneccl` links the `nccl` module instead,
  and `bootstrap.sh oneccl-nccl` refuses to run while the variable is set rather
  than guess which NCCL it should resolve.

### ⚠️ Recipe B: swap a pinned git ref

oneCCL and OSHMPI are built from a branch, and both refs install to the same
prefix and write to the same `results/`. Force the rebuild and separate the
results yourself:

```bash
export GPU_BENCH_ONECCL_OSHMPI_REF=my-branch          # or _NCCL_REF
GPU_BENCH_FORCE=1 make bootstrap TARGETS=oneccl-oshmpi

GPU_BENCH_RESULTS_ROOT=results-my-branch \
  cluster/harness/launch.sh --all halo_1d
```

Skip `GPU_BENCH_FORCE=1` and the bootstrap sees an installed prefix and does
nothing. Skip `GPU_BENCH_RESULTS_ROOT` and benchscribe averages both refs into
one cell.

### 🔒 Recipe C: swap a module-provided library

MPI comes from the site's modules, as does the NCCL the sycl stack links, so
changing one means editing the `module load` line in `env/cuda.sh` or
`env/sycl.sh`. That changes the toolchain for **every backend on that stack**,
so rebuild everything and treat the whole results tree as a separate
experiment:

```bash
GPU_BENCH_RESULTS_ROOT=results-mpi-2.21 \
  cluster/harness/launch.sh --all allreduce
```

This is not a per-measurement knob. If you want an A/B, give the library the
Recipe A treatment instead: add a `deps/<lib>.sh` and one
`gpu_bench_select_variant <lib>` call in `layout.sh`. Build tree, results tree
and `--explain` output already follow from `GPU_BENCH_VARIANT_TAG`. NCCL was
moved out of this recipe that way, and `deps/nccl.sh` is the shorter of the two
worked examples.

## ▶️ Run

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

## 🗂️ Leonardo Layout

| Path | Responsibility |
| --- | --- |
| `cluster.sh` | Interface used by the shared harness |
| `backends.sh` | Backend, launcher, preset, and binary registry |
| `slurm.sh` | Partition and optional account settings |
| `layout.sh` | Dependency source and installation prefixes, library version selection |
| `env/` | Build toolchains and modules |
| `runtime/` | UCX, UCC, NCCL, NVSHMEM, MPI, and oneCCL settings |
| `deps/` | Third-party dependency build targets |
| `bootstrap.sh` | Dependency build entry point |

Machine-independent experiment semantics belong under `cluster/harness`, not in
this directory.
