# Analysis Tools

| Tool | Purpose | Requirements |
| --- | --- | --- |
| [`benchscribe`](benchscribe/README.md) | Parse, aggregate, compare, and characterize benchmark results | Python 3.10+, standard library only |
| [`gpu-bench-plot`](plot/README.md) | Draw latency, bandwidth, fit, distribution, and speedup figures | Python 3.10+, `uv` |
| [`roofline`](roofline/README.md) | Plot kernel rooflines from Nsight Compute CSV exports | Python 3.10+, `uv` |

## Results Workflow

Run Benchscribe from the repository root, then pass its versioned JSON to the
plot package:

```bash
python3 tools/benchscribe results --format json > points.json
python3 tools/benchscribe results --fit --format json > fit.json

uv run --project tools/plot gpu-bench-plot \
  --points points.json --fit fit.json \
  --benchmark halo_1d --figure all --outdir figures
```

Benchscribe is the only component that parses job output and computes the
latency/bandwidth characterization. Plotters consume its JSON rather than
reimplementing those rules.

To exercise the pipeline without a cluster, generate explicitly synthetic data:

```bash
uv run --project tools/plot gpu-bench-plot-sample /tmp/demo-results
python3 tools/benchscribe /tmp/demo-results --format json > /tmp/points.json
uv run --project tools/plot gpu-bench-plot \
  --points /tmp/points.json --outdir /tmp/figures
```

Synthetic results are for tool testing only and must not be published as
measurements.
