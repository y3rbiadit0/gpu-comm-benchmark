# Contributing

Contributions should preserve the central comparison rule: implementations of a
benchmark must measure equivalent GPU-resident work, while backend-specific
completion and synchronization costs remain visible.

## Development Setup

CPU-only tests require CMake 3.24 or newer, a C++17 compiler, and Python 3.10 or
newer. Backend builds additionally require their CUDA or SYCL compiler, MPI, and
communication library. Leonardo users should follow the
[cluster guide](cluster/leonardo/README.md).

Configure and run the common C++ test without enabling a GPU backend:

```bash
cmake -S . -B build/tests -DBUILD_TESTING=ON
cmake --build build/tests
ctest --test-dir build/tests --output-on-failure
```

Run the Python tool tests and checks with:

```bash
python3 -m unittest discover -s tools/benchscribe/tests
uv run --project tools/plot pytest tools/plot/tests
uv run --project tools/plot ruff check tools/plot
```

## Making Changes

- Keep shared timing, reporting, validation, and statistics behavior under
  `include/`; do not duplicate it in individual backends.
- Add benchmark semantics and defaults under
  `cluster/harness/experiments/<benchmark>`.
- Keep compiler modules, library paths, launchers, and transport tuning under
  `cluster/<name>`.
- Preserve the output contract in [`docs/output-schema.md`](docs/output-schema.md).
  Incompatible Benchscribe JSON changes require a schema-version update and a
  matching plot-reader update.
- Do not commit generated builds, benchmark output, profiler reports, or
  synthetic sample data.

## Adding A Benchmark

1. Implement the same operation and validation rule for each applicable backend
   under `src/<model>/<backend>`.
2. Add the benchmark to `cluster/harness/matrix.sh` with its valid backends and
   topologies.
3. Add `cluster/harness/experiments/<benchmark>/common.sh` for arguments and
   defaults, plus a concise README describing the measured operation.
4. Confirm that every implementation emits the standardized report fields and
   handles unsupported capabilities explicitly.
5. Add focused tests for shared parsing, reporting, or statistical behavior.

## Adding A Backend Or Cluster

A backend is registered by the cluster because its compiler, launcher, and
runtime transport are machine-specific. Add its preset and implementation, then
add one backend row, environment, and runtime definition under each supporting
cluster.

To add a cluster, implement the interface documented in
[`cluster/harness/README.md`](cluster/harness/README.md#adding-a-cluster). Shared
experiment files must not name cluster modules, partitions, install prefixes, or
network transports.

## Pull Requests

Describe which benchmark/backend/topology combinations changed and how they were
validated. Include the commands used for tests. Performance changes should state
the machine, software stack, topology, problem size, and number of independent
jobs; do not use profiled runs as benchmark results.
