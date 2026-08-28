# Experiment Harness

The harness turns a `(benchmark, backend, topology)` cell into a scheduler job.
It owns experiment defaults, benchmark-level collective policy, and result
collection but contains no compiler modules, scheduler account, install prefix,
or machine-specific transport tuning.

## Launch

Submit one cell, a benchmark subset, or the complete declared matrix:

```bash
cluster/harness/launch.sh allreduce cuda_mpi 1n4g
cluster/harness/launch.sh --all allreduce alltoall
cluster/harness/launch.sh --all
```

Inspect a cell or the matrix without submission:

```bash
cluster/harness/launch.sh --explain allreduce cuda_mpi 1n4g
cluster/harness/launch.sh --dry-run --all
```

The selected cluster defaults to Leonardo. Override it for another integration:

```bash
GPU_BENCH_CLUSTER=<name> cluster/harness/launch.sh --dry-run --all
```

The topology syntax is `<nodes>n<gpus-per-node>g`; one rank drives one GPU.
`matrix.sh` defines the valid benchmark, backend, and topology combinations.

## Experiment operations

| Benchmark | Operational guide | Contract |
| --- | --- | --- |
| `pingpong` | [`experiments/pingpong`](experiments/pingpong/README.md) | [`docs/benchmarks/pingpong.md`](../../docs/benchmarks/pingpong.md) |
| `halo_1d` | [`experiments/halo_1d`](experiments/halo_1d/README.md) | [`docs/benchmarks/halo-1d.md`](../../docs/benchmarks/halo-1d.md) |
| `allreduce` | [`experiments/allreduce`](experiments/allreduce/README.md) | [`docs/benchmarks/allreduce.md`](../../docs/benchmarks/allreduce.md) |
| `alltoall` | [`experiments/alltoall`](experiments/alltoall/README.md) | [`docs/benchmarks/alltoall.md`](../../docs/benchmarks/alltoall.md) |
| `cg_step` | [`experiments/cg_step`](experiments/cg_step/README.md) | [`docs/benchmarks/cg-step.md`](../../docs/benchmarks/cg-step.md) |
| `moe` | [`experiments/moe`](experiments/moe/README.md) | [`docs/benchmarks/moe.md`](../../docs/benchmarks/moe.md) |

Contracts define benchmark semantics, timing, validation, metrics, and
command-line meaning. These operational guides repeat the invocation syntax for
convenience and define harness defaults, topology constraints, overrides, and
launch examples. Library mappings and build instructions are in the
[backend implementations](../../src/README.md).

## Overrides

Experiment defaults are environment variables inherited by each job. Common
overrides include:

| Variable | Purpose |
| --- | --- |
| `GPU_BENCH_N` | Maximum message elements or application problem size |
| `GPU_BENCH_ITERS` | Timed iterations |
| `GPU_BENCH_WARMUP` | Untimed warmup iterations |
| `GPU_BENCH_NTRIALS` | Trials inside one allocation; default `3` |
| `GPU_BENCH_REPEATS` | Independent jobs per matrix cell |
| `GPU_BENCH_MSG_SIZES` | Comma-separated message sizes for sweep benchmarks |
| `GPU_BENCH_RESULT_NAME` | Result-tree name |
| `GPU_BENCH_TRIAL_TIMEOUT` | Per-trial timeout; default `2m` |
| `GPU_BENCH_ONLY_BACKENDS` | Space-separated backend filters for `--all` |
| `GPU_BENCH_ONLY_TOPOS` | Space-separated topology filters for `--all` |

Benchmark-specific options are documented under [`experiments/`](experiments/).
For example:

```bash
GPU_BENCH_MSG_SIZES=1,1024 GPU_BENCH_NTRIALS=1 \
  cluster/harness/launch.sh allreduce cuda_mpi 1n4g
```

## Results and profiling

Each trial writes separate standard output and error files:

```text
results/<result-name>/<benchmark>/<job-name>-<job-id>-<trial>-stdout.txt
results/<result-name>/<benchmark>/<job-name>-<job-id>-<trial>-stderr.txt
```

`GPU_BENCH_PROFILE=nsys` enables Nsight Systems. `GPU_BENCH_PROFILE=ncu` enables
Nsight Compute with the roofline set. Profiling perturbs timing, so use a
dedicated one-trial result name and never include profiled output in performance
summaries.

## Structure

| Path | Responsibility |
| --- | --- |
| `launch.sh` | Validate and submit one or many cells |
| `job.sh` | Resolve the selected cell on a compute node |
| `matrix.sh` | Declare benchmark, backend, and topology coverage |
| `experiments/common.sh` | Shared execution, profiling, timeout, and result handling |
| `experiments/<benchmark>/common.sh` | Benchmark arguments and defaults |
| `utils/` | Rank wrapping and environment diagnostics |

The harness reaches machine-specific behavior only through
`cluster/<name>/cluster.sh`.

## Adding a cluster

Create `cluster/<name>/cluster.sh` and the environment/runtime files it dispatches
to. The cluster interface provides:

| Function or variable | Purpose |
| --- | --- |
| `SBATCH_PARTITION` | Default scheduler partition |
| `SBATCH_ACCOUNT` | Optional scheduler account |
| `gpu_bench_backend_fields <backend>` | Resolve stack, runtime, launcher, preset, binary directory, and target prefix |
| `gpu_bench_backend_names` | List available backends |
| `gpu_bench_walltime_for <nodes>` | Return scheduler wall time |
| `gpu_bench_cluster_environment <stack>` | Load build/runtime modules and compiler paths |
| `gpu_bench_cluster_runtime <runtime>` | Apply communication-library settings |
| `gpu_bench_cluster_env_files <stack> <runtime>` | List those files for `--explain` |

Keep build-time compiler setup under `env/` and run-time communication tuning
under `runtime/`. Adding a cluster must not require changes under `harness/`.
