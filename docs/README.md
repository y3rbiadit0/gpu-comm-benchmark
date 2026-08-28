# Documentation

This directory contains project-wide benchmark contracts, design rationale, and
analysis methodology. Operational instructions remain beside the cluster,
experiment, backend, and tool code they describe.

## Benchmark contracts

- [`benchmarks/pingpong.md`](benchmarks/pingpong.md): two-endpoint round-trip
  operation and one-way reporting convention
- [`benchmarks/halo-1d.md`](benchmarks/halo-1d.md): periodic neighbor exchange,
  timing cases, and aggregate traffic convention
- [`benchmarks/allreduce.md`](benchmarks/allreduce.md): float sum and normalized
  collective bus bandwidth
- [`benchmarks/alltoall.md`](benchmarks/alltoall.md): personalized exchange,
  permutation validation, and per-peer sizing
- [`benchmarks/cg-step.md`](benchmarks/cg-step.md): stencil halo and two-reduction
  application pattern
- [`benchmarks/moe.md`](benchmarks/moe.md): variable-count dispatch and combine
  application pattern

These contracts own operation semantics, timing boundaries, validation, and
reported metrics, including command-line syntax and argument meaning. Harness
defaults, environment overrides, topology constraints, and launch examples are
documented under
[experiment operations](../cluster/harness/README.md#experiment-operations).
Backend/library mappings and build instructions remain with the
[implementations](../src/README.md).

Contract filenames use kebab-case for readable documentation paths. Executable,
harness, and result identifiers retain their source names, such as `halo_1d` and
`cg_step`.

## Design rationale

- [`design/benchmark-selection.md`](design/benchmark-selection.md): benchmark
  selection, classification, and relation to OSU Micro-Benchmarks

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
- [`src/`](../src/README.md): backend-specific implementation and build notes
- [`tools/`](../tools/README.md): result processing and plotting
