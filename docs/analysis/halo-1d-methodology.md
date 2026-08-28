# Halo 1D Analysis Methodology

This guide defines a reproducible method for identifying the message-size
regimes in which each halo-exchange backend is latency or bandwidth limited. It
does not embed results from a particular software revision.

## Descriptive Model

For aggregate bytes `m`, the Hockney form is:

```text
T(m) = alpha + m / B_inf
```

`alpha` represents fixed submission, matching, launch, progress, and completion
costs. `B_inf` represents the large-message bus-bandwidth ceiling. Their product
estimates the half-bandwidth point:

```text
n_half = alpha * B_inf
```

Benchscribe reports descriptive anchors rather than a global linear regression:

- `alpha`: median measured latency for messages up to 4 KiB;
- `B_inf`: best measured bandwidth;
- `n_half`: their product;
- `peak_at`: message size at peak bandwidth; and
- `tail`: bandwidth at the largest measured message.

Protocol changes and bandwidth cliffs can make one global linear fit misleading.
Always inspect the full latency and bandwidth curves alongside these summaries.

Halo bandwidth counts two sends and two receives. Its `alpha` and `B_inf`
describe a concurrent exchange, not a single one-way message, and should not be
compared directly with published ping-pong parameters.

## Comparison Method

1. Run the same size list for every backend and topology.
2. Keep `isolated` and `steady` cases separate.
3. Compare small-message `usec` to identify the latency floor.
4. Compare `gbytes_per_s`, `peak_at`, and `tail` to identify saturation and
   large-message regressions.
5. Locate crossovers from the plotted curves, not from backend-wide averages.
6. Repeat cells as independent jobs so allocation-to-allocation spread is visible.

The cleanest model comparisons share a communication style:

| Pair | Shared property | Remaining difference |
| --- | --- | --- |
| `cuda_mpi` / `sycl_mpi` | Persistent two-sided MPI | CUDA versus SYCL buffer and kernel setup |
| `cuda_nvshmem` / `oshmpi` | One-sided remote writes | Device signals versus host issue and barrier |
| `cuda_nccl` / `sycl_oneccl` | Grouped library point-to-point | Library and transport implementation |

On Leonardo, CUDA MPI and SYCL MPI both use HPC-X 2.19. Inter-node differences
therefore should not be attributed to different MPI distributions, but compiler,
buffer, runtime, and measurement details still need consideration.

## Collect Timing Data

Use a dedicated result name and independent job repeats:

```bash
export GPU_BENCH_RESULT_NAME=halo-crossover
export GPU_BENCH_REPEATS=5
export GPU_BENCH_MSG_SIZES=1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576

cluster/harness/launch.sh --all halo_1d
```

Summarize and plot the completed result tree:

```bash
python3 tools/benchscribe results/halo-crossover --format json > points.json
python3 tools/benchscribe results/halo-crossover --fit --format json > fit.json
uv run --project tools/plot gpu-bench-plot \
  --points points.json --fit fit.json \
  --benchmark halo_1d --figure all --outdir figures
```

## Attribute Latency

Wall-clock data identifies a crossover; a timeline helps explain it. Profile a
separate one-trial job at representative small and large sizes:

```bash
GPU_BENCH_PROFILE=nsys GPU_BENCH_NTRIALS=1 \
GPU_BENCH_RESULT_NAME=halo-profile \
GPU_BENCH_MSG_SIZES=16,1048576 \
  cluster/harness/launch.sh halo_1d cuda_nvshmem 2n4g
```

Use the same topology and size list for comparison backends. In Nsight Systems,
inspect:

| Backend | Expected critical-path components |
| --- | --- |
| MPI | Host request start/wait, transport progress, data transfer |
| NCCL | Host enqueue, device kernel launch, proxy progress, transfer |
| NVSHMEM | Cooperative launch, device issue, signal dependency |
| OSHMPI | Host put issue, quiet, CUDA completion, barrier |

Profiling changes timing. Use it only for attribution and never merge profiled
records into the timing result tree.
