# GPU Communication Benchmark

GPU Communication Benchmark compares communication models with equivalent
GPU-resident workloads. The suite runs the same communication patterns through
MPI, collective libraries, and one-sided SHMEM implementations so their latency,
bandwidth, scaling, and application-level behavior can be compared directly.

The project currently targets NVIDIA GPUs and is validated on the Leonardo
supercomputer. CUDA and SYCL implementations are provided where the underlying
libraries support both programming models.

## Benchmarks

| Benchmark | Kind | Measures |
| --- | --- | --- |
| [`pingpong`](cluster/harness/experiments/pingpong/README.md) | Microbenchmark | Two-endpoint one-way latency and bandwidth |
| [`halo_1d`](docs/halo_1d.md) | Microbenchmark | Neighbor exchange in a periodic ring |
| [`allreduce`](cluster/harness/experiments/allreduce/README.md) | Microbenchmark | Collective sum latency and bandwidth |
| [`alltoall`](cluster/harness/experiments/alltoall/README.md) | Microbenchmark | Personalized exchange and bus bandwidth |
| [`cg_step`](cluster/harness/experiments/cg_step/README.md) | Application pattern | SpMV halo exchange followed by two reductions |
| [`moe`](cluster/harness/experiments/moe/README.md) | Application pattern | Variable, skewed mixture-of-experts traffic |

Microbenchmarks sweep message size. Application patterns use a fixed problem
size and sweep rank count, testing whether isolated communication results predict
the behavior of a mixed workload. The rationale for this suite is documented in
[`docs/experiments_considerations.md`](docs/experiments_considerations.md).

## Implementations

| Model | Backend | Implementation |
| --- | --- | --- |
| MPI | CUDA-aware MPI | [`src/mpi/cuda`](src/mpi/cuda/README.md) |
| MPI | SYCL + CUDA-aware MPI | [`src/mpi/sycl`](src/mpi/sycl/README.md) |
| XCCL | NCCL | [`src/xccl/cuda`](src/xccl/cuda/README.md) |
| XCCL | oneCCL | [`src/xccl/sycl`](src/xccl/sycl/README.md) |
| SHMEM | NVSHMEM | [`src/shmem/nvshmem`](src/shmem/nvshmem/README.md) |
| SHMEM | OSHMPI | [`src/shmem/oshmpi`](src/shmem/oshmpi/README.md) |

See the [support matrix](docs/support-matrix.md) for benchmark, backend, and
topology coverage.

## Build

### Build DPC++

The SYCL backends require an
[intel/llvm source build with NVIDIA CUDA support](https://intel.github.io/llvm/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda).
With hwloc installed under `$HOME/local/hwloc`, load the Leonardo build modules
and build DPC++ 6.3 once:

```bash
module load gcc/12.2.0 cmake/4.1.2 ninja python cuda/12.2

export DPCPP_HOME="$HOME/opt/dpcpp"
export HWLOC_ROOT="$HOME/local/hwloc"
mkdir -p "$DPCPP_HOME"
git clone https://github.com/intel/llvm -b v6.3.0 --depth=1 \
  "$DPCPP_HOME/llvm"

CC=gcc CXX=g++ python "$DPCPP_HOME/llvm/buildbot/configure.py" \
  --cuda \
  -DCUDA_Toolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_PREFIX_PATH="$HWLOC_ROOT" \
  -DLIBHWLOC_INCLUDE_DIRS="$HWLOC_ROOT/include" \
  -DLIBHWLOC_LIBRARIES="$HWLOC_ROOT/lib/libhwloc.so"

CC=gcc CXX=g++ python "$DPCPP_HOME/llvm/buildbot/compile.py"
```

### Initialize The Project

With `DPCPP_HOME` and `HWLOC_ROOT` still exported, initialize the dependency
stack and build every benchmark preset with:

```bash
make init
```

For prerequisite paths, dependency targets, and validated toolchains, follow the
[Leonardo build guide](cluster/leonardo/README.md#build).

## Run

Machine-independent experiment definitions live in `cluster/harness`; machine
configuration lives in `cluster/<name>`. Exact setup and submission commands
belong to each cluster guide:

| Cluster | Status | Guide |
| --- | --- | --- |
| Leonardo | Validated | [`cluster/leonardo/README.md`](cluster/leonardo/README.md) |

The shared launcher accepts a benchmark, backend, and topology:

```bash
cluster/harness/launch.sh --explain allreduce cuda_mpi 1n4g
cluster/harness/launch.sh allreduce cuda_mpi 1n4g
```

`--explain` resolves a cell without submitting it. See the
[harness guide](cluster/harness/README.md) for matrix runs, overrides, result
paths, and adding another cluster.

## Analyze Results

Benchmark processes emit standardized `key=value` records. Benchscribe collects
records from `results/`, and the plot package consumes its JSON output:

```bash
python3 tools/benchscribe --benchmark allreduce
python3 tools/benchscribe --format json > points.json
uv run --project tools/plot gpu-bench-plot \
  --points points.json --benchmark allreduce --outdir figures
```

See [`tools/README.md`](tools/README.md) for the analysis workflow and
[`docs/output-schema.md`](docs/output-schema.md) for the report contract.

## Project Documentation

- [`docs/README.md`](docs/README.md): benchmark design and analysis notes
- [`cluster/README.md`](cluster/README.md): cluster integration and supported systems
- [`CONTRIBUTING.md`](CONTRIBUTING.md): development and contribution workflow
- [`LICENSE`](LICENSE): MIT license
