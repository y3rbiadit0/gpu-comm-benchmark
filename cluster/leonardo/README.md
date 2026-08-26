# Leonardo

Leonardo uses different validated software stacks for native CUDA and SYCL-on-NVIDIA runs. Select the stack explicitly before configuring or submitting jobs.

## Setup

From a clean clone, build the external libraries before any benchmark:

```bash
./cluster/leonardo/bootstrap.sh
```

It resolves target order itself and builds patched OSHMPI, then oneCCL with the OSHMPI
backend, then the `leonardo-sycl-oneccl-oshmpi` benchmarks. Targets live in
[`deps/`](deps/); each declares the stack it needs, what it requires, and a path that
proves it is already built, so re-running skips finished work.

```bash
./cluster/leonardo/bootstrap.sh --list           # available targets
./cluster/leonardo/bootstrap.sh oneccl-oshmpi    # one target and its dependencies
GPU_BENCH_FORCE=1 ./cluster/leonardo/bootstrap.sh oneccl-oshmpi   # rebuild anyway
```

The only prerequisite it cannot install is a **DPC++ compiler**. Point `DPCPP_HOME` at
one (default `$HOME/opt/dpcpp_6.3`).

### Where things go

[`layout.sh`](layout.sh) is the single definition of every path the bootstrap produces
and every path the runtime scripts consume, so the two cannot drift:

| Variable | Default | Holds |
| --- | --- | --- |
| `GPU_BENCH_WORK_ROOT` | `$SCRATCH/gpu-comm-bench` | clones and build trees - large, disposable |
| `GPU_BENCH_PREFIX_ROOT` | `$HOME/opt/gpu-comm-bench` | install prefixes - small, persistent, resolved by jobs at run time |

Relocate everything by setting one of those. Individual prefixes (`OSHMPI_HOME`,
`ONECCL_OSHMPI_ROOT`, `ONECCL_NCCL_ROOT`) are respected if already exported, for one-off
experiments. Nothing else defines them - two definitions with different defaults would
resolve by source order, which is how a build ends up linking one install's headers
against another's libraries.

`environment.sh` sources `layout.sh`, so `make leonardo` and the bootstrap agree on
every path.

### Not yet bootstrapped

`leonardo-sycl-oneccl` needs oneCCL built with the **NCCL** backend at
`$ONECCL_NCCL_ROOT`. There is no `deps/` target for it yet; it is hand-built, and
`layout.sh` defaults to where that install already lives. See
[`../../oneccl-unisa.md`](../../oneccl-unisa.md).

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


## How experiments are defined

One job script, `cluster/leonardo/experiments/job.sh`, serves every
(benchmark, backend, topology) cell. It replaced 230 near-identical per-cell
scripts that differed only in values derivable from those three names.

| file | holds |
| --- | --- |
| `experiments/job.sh` | the submitted script: resolve, validate, dispatch |
| `experiments/backends.sh` | per-backend constants (stack, runtime, launcher, preset, binary) |
| `experiments/matrix.sh` | which cells are valid, and why each exclusion exists |
| `experiments/<bench>/common.sh` | per-benchmark defaults and arguments |

The allocation shape is not baked into a file: `--nodes`, `--ntasks-per-node`,
`--gres`, `--time` and `--job-name` are passed on the sbatch command line, which
takes precedence over `#SBATCH` directives. Nothing is generated, so there is
nothing to regenerate or keep in sync.

Adding a topology is a one-line edit in `matrix.sh`. Adding a backend is one row
in `backends.sh` plus its name in the benchmarks that support it.

## Experiments

Experiment scripts are fixed-topology Slurm launchers. They write Slurm logs under `logs/` and benchmark outputs under `results/<result-name>/<problem>/`.
The active suite is `pingpong`, `halo_1d`, `allreduce`, `alltoall`, `cg_step`, and `moe`.

Halo 1D:

```bash
tools/launch.sh halo_1d cuda_mpi 1n4g
tools/launch.sh halo_1d cuda_nccl 1n4g
tools/launch.sh halo_1d cuda_nvshmem 1n4g
tools/launch.sh halo_1d oshmpi 1n4g
tools/launch.sh halo_1d sycl_mpi 1n4g
tools/launch.sh halo_1d sycl_oneccl 1n4g
```

Allreduce (collective sum latency/bandwidth, internal size sweep):

```bash
tools/launch.sh allreduce cuda_mpi 1n4g
tools/launch.sh allreduce cuda_nccl 1n4g
tools/launch.sh allreduce cuda_nvshmem 1n4g
tools/launch.sh allreduce oshmpi 1n4g
tools/launch.sh allreduce sycl_mpi 1n4g
tools/launch.sh allreduce sycl_oneccl 1n4g
```

Each `allreduce` backend also has `1n1g`, `1n2g` (NVLink), `2n1g` (InfiniBand), and `2n4g`
launchers; see [`experiments/allreduce/README.md`](experiments/allreduce/README.md).

Ping-pong (point-to-point one-way latency/bandwidth, 2 endpoints, internal size sweep):

```bash
tools/launch.sh pingpong cuda_mpi 1n2g   # NVLink
tools/launch.sh pingpong cuda_mpi 2n1g   # InfiniBand
tools/launch.sh pingpong cuda_nccl 2n1g
tools/launch.sh pingpong cuda_nvshmem 2n1g
tools/launch.sh pingpong oshmpi 2n1g
tools/launch.sh pingpong sycl_mpi 2n1g
tools/launch.sh pingpong sycl_oneccl 2n1g
```

Only the `1n2g` (intra-node NVLink) and `2n1g` (inter-node InfiniBand) topologies exist for
`pingpong`; see [`experiments/pingpong/README.md`](experiments/pingpong/README.md).

All-to-all (personalized exchange / bisection bandwidth):

```bash
tools/launch.sh alltoall cuda_mpi 1n4g
tools/launch.sh alltoall cuda_nccl 1n4g
tools/launch.sh alltoall cuda_nvshmem 1n4g
tools/launch.sh alltoall oshmpi 1n4g
tools/launch.sh alltoall sycl_mpi 1n4g
tools/launch.sh alltoall sycl_oneccl 1n4g
```

Each `alltoall` backend has `1n1g`, `1n2g`, `1n4g`, `2n1g`, and `2n4g` launchers; see
[`experiments/alltoall/README.md`](experiments/alltoall/README.md).

MoE (top-1 variable-count dispatch + combine under uniform, local, and hotspot routing):

```bash
tools/launch.sh moe cuda_mpi 1n4g
tools/launch.sh moe cuda_nccl 1n4g
tools/launch.sh moe cuda_nvshmem 1n4g
tools/launch.sh moe oshmpi 1n4g
tools/launch.sh moe sycl_mpi 1n4g
tools/launch.sh moe sycl_oneccl 1n4g
```

MoE is a skew-sensitive global personalized application pattern: unlike dense `alltoall`,
its per-peer operation sizes and expert receive loads vary with routing. Each backend has
`1n1g`, `1n2g`, `1n4g`, `2n1g`, and `2n4g` launchers; see
[`experiments/moe/README.md`](experiments/moe/README.md). oneCCL launchers use `mpirun`.

CG step (conjugate-gradient iteration skeleton: SpMV halo + two reductions):

```bash
tools/launch.sh cg_step cuda_mpi 1n4g
tools/launch.sh cg_step cuda_nccl 1n4g
tools/launch.sh cg_step cuda_nvshmem 1n4g
tools/launch.sh cg_step oshmpi 1n4g
tools/launch.sh cg_step sycl_mpi 1n4g
tools/launch.sh cg_step sycl_oneccl 1n4g
```

Each `cg_step` backend has `1n1g`, `1n2g`, `1n4g`, `2n1g`, and `2n4g` launchers; see
[`experiments/cg_step/README.md`](experiments/cg_step/README.md).

Useful overrides:

```bash
GPU_BENCH_N=16777216 GPU_BENCH_NTRIALS=5 tools/launch.sh halo_1d cuda_mpi 1n4g
GPU_BENCH_RESULT_NAME=halo-sycl-test tools/launch.sh halo_1d sycl_mpi 1n4g
GPU_BENCH_ITERS=500 GPU_BENCH_WARMUP=100 tools/launch.sh allreduce cuda_nccl 2n1g
GPU_BENCH_HIDDEN=512 GPU_BENCH_ROUTINGS=uniform,hotspot80 tools/launch.sh moe cuda_nccl 2n4g
```

Per-problem notes and validated results live in `cluster/leonardo/experiments/<problem>/README.md`.
