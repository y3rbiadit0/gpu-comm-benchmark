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
python3 tools/benchscribe results --benchmark allreduce

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

- **α** — latency floor: the median per-iteration time for messages up to 4 KiB.
  If a truncated sweep starts above 4 KiB, the smallest available point is used.
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
- Groups by **(benchmark, case, topology, problem size `n`)** and aggregates the metric
  across trials (mean + min). For records with `hidden`, the displayed/grouping case is
  `<case>,hidden=<width>` (for example `uniform,hidden=256`), keeping MoE routing
  distributions and hidden widths separate. Records without `hidden` retain their plain case.
- Within each group, reports each backend relative to `cuda_mpi`:
  - **Δ%** — relative difference (negative = faster than baseline).
  - **Speedup** — `cuda_mpi` / backend for latency metrics (>1 = faster).
- Sorts backend rows by speedup descending within each benchmark/topology/size group.
- Parses both the new schema (`usec`, `gbytes_per_s`, …) and legacy `time_s=` lines.
  For latency benchmarks compare `usec`;
  for `pingpong` the `GB/s` column is one-way bandwidth per swept message size.
- Uses only `status=OK validation=PASS` records for metrics, bandwidth,
  baselines, deltas, speedups, and α–β fits. `ERROR`, validation-failed, and
  contradictory `status=OK validation=FAIL` records are excluded from numeric
  results.
- Preserves `status=NOT_IMPLEMENTED validation=SKIP` capability results as
  `N/I`. `NOT_IMPLEMENTED` paired with `PASS` or `FAIL` is contradictory, becomes
  `ERROR`, and is excluded. Mixed OK/N/I groups count only timing-contributing OK
  trials, while N/I-only groups retain their observed trial count.
- Supports `--metric usec`, `--metric time_per_iter_s`, `--metric time_s`, and
  `--metric gbytes_per_s` when you want to force a specific comparison metric.

Only stdlib is required (Python 3.10+). No build step.

### JSON output

`--format json` carries more than the Markdown and CSV views, which collapse
each point to one number. It keeps `runs[]` (each job's within-run iteration
distribution — for box plots) and `across_runs` (spread across independent jobs
— for error bands). `--fit --format json` does the same for the α–β numbers.
Both files carry a `schema_version`; that is the contract `tools/plot` reads
against, so bump it on any incompatible change.

## `sbatch.sh`

Submits a single job with the project's SLURM defaults applied.

```bash
tools/sbatch.sh cluster/leonardo/experiments/allreduce/oshmpi/1n4g.sh
GPU_BENCH_N=17 tools/sbatch.sh cluster/leonardo/experiments/halo_1d/cuda_mpi/2n4g.sh
tools/sbatch.sh --qos=boost_qos_dbg cluster/leonardo/experiments/cg_step/cuda_mpi/2n4g.sh
```

The job scripts carry no `-A`/`-p` directives on purpose, so anyone can point
them at their own allocation — those defaults live in environment variables set
by `cluster/leonardo/slurm.sh`. `tools/submit_all.sh` sources that file
in-process and so always has them; a hand-run `sbatch cluster/.../1n4g.sh` from
a fresh login shell does **not**, and silently lands on the cluster's default
partition (`lrd_all_serial` on Leonardo), which fails with a confusing
`QOSMaxCpuPerUserLimit` or `More processors requested than permitted`.

This wrapper sources the defaults, prints the account and partition it is using,
and passes everything else straight through — so sbatch flags still work and
still win, the precedence being command line > environment > script directives.

Use `tools/submit_all.sh` for whole sweeps; this is for one job at a time.

## `plot_bench`

Draws the figures for the results, straight from Benchscribe JSON. It never
re-parses Slurm output and never re-implements the fit — Benchscribe owns both.

Unlike Benchscribe, this one needs matplotlib, so it is a self-contained `uv`
project (`tools/plot`) with its dependencies declared and locked in its own
`pyproject.toml`/`uv.lock`. Benchscribe stays stdlib-only.

```bash
python3 tools/benchscribe results --format json       > points.json
python3 tools/benchscribe results --fit --format json > fit.json

uv run --project tools/plot gpu-bench-plot \
    --points points.json --fit fit.json \
    --benchmark halo_1d --figure all --outdir docs/analysis/figures
```

`--project tools/plot` leaves the working directory alone, so the paths above
stay relative to the repository root.

| Figure | What it shows |
| --- | --- |
| `sweep` | per-exchange time vs message size, log-log, panel per case × topology |
| `bandwidth` | bus GB/s vs message size, same panel grid |
| `fit` | α, B∞ and n½ as grouped bars, one panel per measure (needs `--fit`) |
| `heatmap` | speedup vs the `cuda_mpi` baseline, backend × message size |
| `dist` | per-job timing distributions at one message size, as box plots (`--size`) |

`--figure` picks one of those or `all`; `--theme light|dark` and
`--format svg|png|pdf` control output; `--size` chooses the message size the
`dist` figure shows.

The `dist` box plots separate the two dispersions Benchscribe tracks: box height
is within-run spread, while the offset between a backend's boxes is spread across
independent jobs — the one that actually speaks to reproducibility.

Every figure writes a companion `.csv` beside it, and colours are fixed per
backend so a hue means the same backend in every figure. Both are contracts, not
conveniences — [`tools/plot/README.md`](plot/README.md) explains why, along with
the rest of the design rules and the schema handshake with Benchscribe.

### Trying it without a cluster

`gpu-bench-plot-sample` writes a synthetic `results/` tree from an α–β model so
the whole pipeline can be exercised offline. Its output is **not measured
data** — never cite or commit it.

```bash
uv run --project tools/plot gpu-bench-plot-sample /tmp/demo-results
python3 tools/benchscribe /tmp/demo-results --format json > /tmp/points.json
uv run --project tools/plot gpu-bench-plot --points /tmp/points.json --outdir /tmp/figures
```
