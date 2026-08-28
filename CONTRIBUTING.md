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
- Define operation semantics, timing, validation, and metrics under
  `docs/benchmarks`, and keep defaults and launch behavior under
  `cluster/harness/experiments/<benchmark>`.
- Keep compiler modules, library paths, launchers, and transport tuning under
  `cluster/<name>`. Benchmark-level collective policy may remain with the shared
  experiment when it is part of the measured configuration.
- Preserve the output contract in
  [`docs/reference/output-schema.md`](docs/reference/output-schema.md).
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
   defaults, plus a concise operational README linking the benchmark contract.
4. Add a kebab-case `docs/benchmarks/<benchmark-doc>.md` defining the operation,
   timing boundary, validation, metrics, and implementation completion
   differences.
5. Add the contract to `docs/README.md` and the root benchmark table, add the
   operational guide to `cluster/harness/README.md`, and update the support
   matrix.
6. Confirm that every implementation emits the standardized report fields and
   handles unsupported capabilities explicitly.
7. Add focused tests for shared parsing, reporting, or statistical behavior.

## Adding A Backend Or Cluster

A backend is registered by the cluster because its compiler, launcher, and
runtime transport are machine-specific. Add its preset and implementation, then
add one backend row, environment, and runtime definition under each supporting
cluster.

To add a cluster, implement the interface documented in
[`cluster/harness/README.md`](cluster/harness/README.md#adding-a-cluster). Shared
experiment files must not name cluster modules, partitions, install prefixes, or
network transports.

## Versioning And Releases

The `VERSION` argument to `project()` in the top-level `CMakeLists.txt` is the
single source of truth for the benchmark suite version. CMake exposes it as
`PROJECT_VERSION` and embeds it in benchmark output. Do not duplicate the suite
version in another source file.

Before 1.0, increment the version as follows:

- Patch (`0.1.0` to `0.1.1`) for compatible fixes. Correctness fixes that change
  numerical results must call that out in the changelog.
- Minor (`0.1.0` to `0.2.0`) for new benchmarks or backends and for intentional
  changes to measurement semantics, command-line interfaces, or output behavior.
- Major (`1.0.0`) when the suite's measurement and user-facing contracts are
  considered stable. After 1.0, incompatible changes increment the major version,
  additive changes increment the minor version, and compatible fixes increment
  the patch version.

The suite version is independent of the Benchscribe JSON `schema_version`, which
changes only when that data contract becomes incompatible. The
`gpu-bench-plot` package in `tools/plot/pyproject.toml` also has its own package
version and changes only when that package is released.

To make a release:

1. Update the top-level `project(... VERSION <version> ...)` declaration and move
   the relevant `CHANGELOG.md` entries from `[Unreleased]` to
   `[<version>] - YYYY-MM-DD`.
2. Run the CPU test commands above and all affected GPU build and benchmark
   presets.
3. Commit the release changes and create an annotated tag whose name exactly
   matches the project version, for example
   `git tag -a 0.2.0 -m "Release 0.2.0"`.
4. Build release binaries from the tag. Their records will contain both
   `suite_version=<version>` and `source_revision=<tag>`.

Normal Git checkouts derive `source_revision` from `git describe`, including a
`-dirty` suffix for uncommitted changes. Source archives without Git metadata use
`unknown`; packagers can set `-DGPU_BENCH_SOURCE_REVISION=<revision>` explicitly.

## Pull Requests

Describe which benchmark/backend/topology combinations changed and how they were
validated. Include the commands used for tests. Performance changes should state
the machine, software stack, topology, problem size, and number of independent
jobs; do not use profiled runs as benchmark results.
