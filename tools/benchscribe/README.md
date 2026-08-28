# Benchscribe

Benchscribe scans benchmark stdout files, parses standardized `key=value`
records, aggregates repeated runs, and compares each backend with the
`cuda_mpi` baseline. It requires only Python 3.10 or newer and the standard
library.

## Use

```bash
python3 tools/benchscribe
python3 tools/benchscribe results --benchmark allreduce
python3 tools/benchscribe --format csv > summary.csv
python3 tools/benchscribe --metric gbytes_per_s --format csv > bandwidth.csv
python3 tools/benchscribe results --benchmark halo_1d --fit
```

Available output formats are Markdown, CSV, and JSON. The default results path is
`results/`; the default format is Markdown.

## Aggregation

Benchscribe:

- scans `**/*-stdout.txt` below the selected results directory;
- recovers topology from the result path;
- groups by benchmark, case, topology, and problem size;
- keeps MoE hidden width as part of the displayed case;
- averages valid trials and reports each backend relative to `cuda_mpi`;
- uses `cuda_mpi / backend` for latency speedup, where values above one are
  faster;
- excludes errors and validation failures from numeric summaries; and
- preserves `NOT_IMPLEMENTED`/`SKIP` capability records as `N/I`.

The process record contract is documented in
[`docs/output-schema.md`](../../docs/output-schema.md).

## Characterization

For message-size sweeps, `--fit` reduces each backend curve to descriptive
latency/bandwidth values without regression:

| Value | Definition |
| --- | --- |
| `alpha` | Median latency for messages up to 4 KiB, or the smallest available point |
| `B_inf` | Best reported bandwidth |
| `n_half` | `alpha * B_inf`, the estimated half-bandwidth message size |
| `peak_at` | Message size at peak bandwidth |
| `tail` | Bandwidth at the largest message size |

Bandwidth keeps each benchmark's convention. Ping-pong reports one-way
bandwidth; halo reports aggregate exchange bus bandwidth. Compare only like
conventions.

## JSON Contract

JSON retains per-job iteration distributions in `runs[]` and independent-job
spread in `across_runs`. Fit JSON carries the corresponding characterization
data. Both documents include `schema_version`, which is checked by
`gpu-bench-plot`; incompatible changes require coordinated version updates.

## Test

```bash
python3 -m unittest discover -s tools/benchscribe/tests
```
