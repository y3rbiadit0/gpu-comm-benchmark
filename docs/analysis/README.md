# Analysis

Authored analysis documents live in this directory:

- [`hockney-model.md`](hockney-model.md) explains why the suite uses a simple
  latency/bandwidth model and where that model stops being reliable.
- [`halo-1d-methodology.md`](halo-1d-methodology.md) defines a reproducible
  workflow for comparing halo-exchange backends and attributing latency.

## Local Artifacts

`data/` contains locally generated Benchscribe JSON, CSV tables, and SVG figures.
These artifacts are ignored by Git and are not available in a clean checkout.
They remain under `docs/analysis/` so a local analysis workspace can keep its
inputs and figures near the corresponding methodology without presenting them as
published repository content.

Generate a `halo_1d` artifact set from the repository root with:

```bash
mkdir -p docs/analysis/data/microbenchmarks/tuned/halo_1d/figures
python3 tools/benchscribe results --format json \
  > docs/analysis/data/microbenchmarks/tuned/halo_1d/points.json
python3 tools/benchscribe results --fit --format json \
  > docs/analysis/data/microbenchmarks/tuned/halo_1d/fit.json
uv run --project tools/plot gpu-bench-plot \
  --points docs/analysis/data/microbenchmarks/tuned/halo_1d/points.json \
  --fit docs/analysis/data/microbenchmarks/tuned/halo_1d/fit.json \
  --benchmark halo_1d --figure all \
  --outdir docs/analysis/data/microbenchmarks/tuned/halo_1d/figures
```

See the [analysis tools guide](../../tools/README.md) for the generic workflow.
