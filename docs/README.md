# Documentation

This directory contains project-wide benchmark contracts, design rationale, and
analysis methodology. Operational instructions remain beside the cluster,
experiment, backend, and tool code they describe.

## Benchmark Design

- [`benchmarks/halo-1d.md`](benchmarks/halo-1d.md): complete halo-exchange
  contract and implementation notes
- [`design/benchmark-selection.md`](design/benchmark-selection.md): benchmark
  selection, classification, and relation to OSU Micro-Benchmarks

Benchmark arguments, defaults, and topology constraints are documented under
[`cluster/harness/experiments`](../cluster/harness/experiments/).

## Reference

- [`reference/output-schema.md`](reference/output-schema.md): benchmark process
  output and timing semantics
- [`reference/support-matrix.md`](reference/support-matrix.md): implemented
  backends and declared experiment coverage

## Analysis

- [`analysis/README.md`](analysis/README.md): methodology index and local
  artifact policy
- [`analysis/hockney-model.md`](analysis/hockney-model.md): rationale and limits
  of the Hockney model
- [`analysis/halo-1d-methodology.md`](analysis/halo-1d-methodology.md):
  reproducible halo latency/bandwidth comparison workflow

## Operational Guides

- [`cluster/`](../cluster/README.md): cluster setup and experiment execution
- [`src/`](../src/): backend-specific implementation notes
- [`tools/`](../tools/README.md): result processing and plotting
