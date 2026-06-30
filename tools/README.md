# Tools

## `benchscribe`

Benchscribe is the results processor for the benchmark suite. It scans the `results/`
tree produced by the Leonardo experiment launchers, parses the standardized `key=value`
report lines emitted by each binary, aggregates across trials, and prints a comparison
summary where **every backend is normalized to the `cuda_mpi` baseline**.

```bash
# Markdown summary of everything under results/
python3 tools/benchscribe

# A specific results directory, or a single benchmark
python3 tools/benchscribe results --benchmark dot_product

# Machine-readable CSV (for plotting / spreadsheets)
python3 tools/benchscribe --format csv > summary.csv

# Override the primary metric used for ranking/comparison
python3 tools/benchscribe --metric gbytes_per_s --format csv > bandwidth.csv

# Latency/bandwidth characterization of a message-size sweep
python3 tools/benchscribe results --benchmark halo_1d --fit

# Save a Markdown report
python3 tools/benchscribe > RESULTS.md
```

### `--fit`: latency floor, peak bandwidth, and the knee

For benchmarks that sweep message size (e.g. `halo_1d`, `pingpong`), `--fit`
collapses each backend's whole sweep into the α–β model numbers, read straight off
the curve (no regression):

- **α** — latency floor: the fastest per-iteration time in the sweep (the flat,
  small-message part).
- **B∞** — peak bandwidth: the best `gbytes_per_s` reached.
- **n½ = α·B∞** — the message size at which you hit half of peak bandwidth.
- **peak @** / **tail** — message size at peak bandwidth, and bandwidth at the
  largest message (exposes a large-message plateau/cliff).

Bandwidth is whatever each binary reports: **bus** (send+receive) for `halo_1d`,
**one-way** (½ round-trip) for `pingpong`. α and B∞ inherit that convention, so
compare like with like.

Works with `--format csv`. See
[`docs/analysis/halo_1d-crossover.md`](../docs/analysis/halo_1d-crossover.md) for
the model and how to read the result.

### What it does

- Walks `results/**/*-stdout.txt` and reads each benchmark report line. The line
  format is documented in the [root README](../README.md#benchmark-output-schema).
- Recovers the topology (`1n4g`, `2n1g`, …) from the result path.
- Groups by **(benchmark, topology, problem size `n`)** and aggregates the metric
  across trials (mean + min).
- Within each group, reports each backend relative to `cuda_mpi`:
  - **Δ%** — relative difference (negative = faster than baseline).
  - **Speedup** — `cuda_mpi` / backend for latency metrics (>1 = faster).
- Sorts backend rows by speedup descending within each benchmark/topology/size group.
- Parses both the new schema (`usec`, `gbytes_per_s`, …) and the legacy
  `time_s=` lines (`vector_add`, `halo_1d`). For latency benchmarks compare `usec`;
  for `pingpong` the `GB/s` column is one-way bandwidth per swept message size.
- Supports `--metric usec`, `--metric time_per_iter_s`, `--metric time_s`, and
  `--metric gbytes_per_s` when you want to force a specific comparison metric.

Only stdlib is required (Python 3.10+). No build step.
