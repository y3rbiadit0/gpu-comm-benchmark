# Milestones

Working roadmap for the thesis benchmark suite. Companion docs:
[experiments_considerations.md](experiments_considerations.md) (pattern selection and
classification) and [analysis/halo_1d-crossover.md](analysis/halo_1d-crossover.md)
(template for the model-fitting work in M3/M4).

## M1 — Deep understanding of every experiment

**Owner:** Franco (reading/study milestone, no code changes expected).

For each pattern × backend, be able to explain:

- what exactly sits inside the timed region (and what is deliberately excluded);
- which communication primitive each backend uses and why
  (e.g. grouped `ncclSend/Recv` vs. native `MPI_Alltoall` vs. `nvshmem_float_alltoall`);
- the sync model differences: two-sided message passing vs. one-sided put+signal,
  host- vs. device-initiated, and where each backend pays for progress
  (e.g. OSHMPI's barrier-based sync on inter-node paths);
- how validation proves correctness locally (no gather).

**Done when:** each experiment can be explained at whiteboard level, including the known
caveats (oneCCL fork pt2pt/broadcast limitations, NVSHMEM single-block bandwidth cap,
OSHMPI host-resident scalar).

## M2 — Environment bootstrap ("init") tool

**Goal:** one command that takes a fresh checkout to a fully built, runnable suite —
cloning, patching, and compiling the dependency stack (oneCCL fork, OSHMPI, etc.) and the
suite itself — for a named environment profile (`leonardo`, `local`, …).

**Recommended shape:** split *bootstrap* from *activation*, because they have different
constraints:

- **Activation stays bash** (`cluster/leonardo/environment.sh` as today): it must be
  *sourced* into the job shell to export modules/paths, which only a shell script can do.
- **Bootstrap becomes `tools/bootstrap.py`**, Python 3 stdlib-only (same constraint that
  `tools/benchscribe` already follows, so it runs on Leonardo login nodes with no pip):
  - `tools/bootstrap.py --env leonardo` / `--env local`
  - driven by a small declarative manifest (TOML/JSON checked into the repo) listing each
    dependency: git URL, pinned ref/commit, cmake/configure flags per environment, and
    install prefix;
  - idempotent: skip a dependency whose pinned ref is already built (stamp file per
    dep+ref), `--force` to rebuild, `--only <dep>` to iterate on one;
  - logs each build to a file and fails loudly with the log path;
  - ends by printing the matching activation line
    (`source cluster/leonardo/environment.sh <stack>`).

  Python over pure bash for this half because the bootstrap is imperative logic —
  per-dependency state, error handling, partial rebuilds — which bash handles poorly, and
  there is nothing to source back into the caller's shell.

**Done when:** on Leonardo, `git clone && tools/bootstrap.py --env leonardo` followed by
one `sbatch` runs a passing experiment; the same flow works for `--env local` (CPU/dev
subset). Pinned refs make results reproducible for the thesis.

**Status (2026-07-07):** scaffolding implemented and mechanics tested locally —
`tools/bootstrap.py` (stdlib-only; clone/build/stamp/idempotency/`--only`/`--force`/
`--list`, per-dep logs, writes `commlibs/env.sh`) + `tools/bootstrap-manifest.json`
(leonardo/local profiles; deps: hwloc, oshmpi, dpcpp, oneccl-nccl). `commlibs/` is
git-ignored; `cluster/leonardo/environment.sh` sources `commlibs/env.sh` before the
env scripts so bootstrapped installs override the `$HOME/opt` fallbacks. Remaining:
fill the oneCCL fork URL (manifest has `FILL_ME`), pin dpcpp/oshmpi refs + confirm
configure flags against the original install notes, then validate a full build on
Leonardo.

## M3 — Roofline model for each experiment kernel

**Goal:** place every device kernel in the suite on the Leonardo A100 roofline
(arithmetic intensity vs. achieved FLOP/s against peak compute and HBM bandwidth).

Scope note: this suite is deliberately comm-dominated, so most timed regions have little
or no kernel — the roofline targets the *compute* components:

- `cg_step`: 5-point stencil kernel + block-reduction dot kernels (the interesting ones —
  they must be fast enough that the step stays comm-bound at the chosen size);
- `halo_1d` NVSHMEM device-initiated kernel (comm kernel — roofline against network, not
  HBM).

`allreduce` has no application compute kernel and is analyzed with the communication
model instead.

Method: `ncu` (Nsight Compute) on single-rank runs for FLOP and DRAM-byte counts →
arithmetic intensity; peaks from A100-64GB specs (or measured via `babelstream`-style
ceilings). One roofline chart, all kernels plotted, in `docs/analysis/`.

**Done when:** the chart exists and the thesis can state, per kernel, "memory-bound at
X% of streaming peak" — closing the loop with the roofline↔Hockney analogy (roofline for
compute, α–β for communication).

## M4 — Hockney (α–β) model results for every pattern

**Goal:** extend the halo_1d crossover analysis to the whole suite.

1. Fit α (latency) and β (inverse bandwidth) per {backend × transport} from the pingpong
   sweep: α from the small-message asymptote, 1/β from the large-message asymptote,
   n½ = α/β as the crossover.
2. Push the fitted parameters through each pattern's cost model
   (see experiments_considerations.md):
   - `halo_1d`: ≈ 2α + 2βn per iteration
   - `allreduce`: algorithm-dependent tree/ring latency plus payload transfer
   - `alltoall`: bisection-limited, per-rank bytes ∝ P
   - `moe`: variable-count dispatch + inverse combine, sensitive to locality and incast
   - `cg_step`: halo term + 2 allreduce terms
3. Compare predicted vs. measured per pattern × backend × topology; report the ratio.
   Agreement validates the model; deviations are findings about the implementation
   (e.g. protocol switches, progress costs, grouped-send overhead).

**Done when:** one table/chart of predicted-vs-measured across the matrix, with each
deviation ≥ ~2× explained or explicitly flagged as open.

## Dependencies and ordering

- M1 has no dependencies and supports everything else — do it while jobs queue.
- M4 needs verified multi-rank results (the current week's run matrix) and pingpong
  sweeps on both transports; it is the analysis half of the results chapter.
- M3 needs `ncu` access on Leonardo and single-rank jobs only — independent of M4,
  can interleave.
- M2 is independent of results; it hardens reproducibility and is the right thing to
  build while waiting on the queue. Not on the critical path for this week's numbers.

## M5 — Swept allreduce

**Goal:** replace scalar-only reduction coverage with a modern tensor allreduce benchmark.

Add a `float32` sum-allreduce message-size sweep to all six backends. The CLI follows the
pingpong convention (`<max_elements> [iterations] [warmup] [message_sizes]`), validates
every output element, and reports both algorithm bandwidth and normalized collective bus
bandwidth. The Leonardo matrix covers `1n1g`, `1n2g`, `1n4g`, `2n1g`, and `2n4g`.

**Done when:** all six targets build, explicit and default size sweeps validate, results are
parsed by Benchscribe, and intra-node, pure inter-node, and mixed runs complete. OSHMPI's
host-symmetric reduction memory is identified in the output.

**Implementation status (2026-07-15):** sources, targets, launchers, reporting, and local
static checks are complete. Leonardo backend builds and topology runs remain.

## M6 — Remove retired patterns

**Goal:** remove benchmarks that no longer contribute results.

Delete all `dot_product` and `vector_add` implementations, including both NVSHMEM
vector-add variants, together with their CMake targets, Leonardo experiment families,
validation helpers, and active documentation references. Keep the two scalar reductions
inside `cg_step`, but describe them directly rather than through the deleted benchmark.

**Done when:** all six backend presets build and a repository search finds no active source,
configuration, launcher, or documentation references to either retired benchmark.

**Implementation status (2026-07-15):** complete; only this historical milestone text
retains the retired names.

## M7 — Correct halo topology coverage

**Goal:** make every halo launch topology valid and comparable.

Remove invalid `1n1g` halo launchers, add `1n2g` for pure intra-node exchange and `2n1g`
for pure inter-node exchange, and retain `1n4g` and `2n4g`. Apply the same valid topology
coverage to the optimized NVSHMEM halo.

**Done when:** no halo launcher starts fewer than two ranks and every supported backend
passes `1n2g`, `1n4g`, `2n1g`, and `2n4g` validation.

**Implementation status (2026-07-15):** launcher and documentation changes are complete;
Leonardo topology validation remains.

## M8 — MoE contract and MPI reference

**Goal:** define one backend-independent variable-token expert exchange.

Use one expert per rank, top-1 routing, `float32` token payloads, and deterministic
`uniform`, `locality80`, and `hotspot80` routing cases. Routing, packing, counts, and
displacements stay outside timing. One timed iteration is a true variable-size dispatch
followed by a variable-size combine. CUDA MPI and SYCL MPI provide the reference
implementations through `MPI_Alltoallv`.

**Done when:** both MPI implementations agree, every token reaches its expert and returns
exactly once, and edge cases including zero-count peers, `tokens < ranks`, uneven counts,
and hotspot routing validate.

**Implementation status (2026-07-15):** shared planning/validation and both MPI references
are complete; host-side plan simulations pass. Accelerator runtime validation remains.

## M9 — MoE backend implementations

**Goal:** port the MPI reference semantics without silently changing the operation.

NCCL uses grouped variable-count `ncclSend`/`ncclRecv`, oneCCL uses variable-count
`ccl::send`/`ccl::recv`, and NVSHMEM/OSHMPI use variable puts into deterministic remote
offsets. No implementation may substitute padded fixed-size alltoall or silently fall back
to MPI.

If the installed oneCCL backend lacks point-to-point support, emit
`status=NOT_IMPLEMENTED reason=point_to_point validation=SKIP`, preserve the result file,
and exclude the entry from performance comparisons. Unexpected oneCCL failures remain
errors. This keeps the missing operation visible as a contribution point.

**Done when:** NCCL, NVSHMEM, and OSHMPI match MPI validation and oneCCL either validates or
reports the standardized unsupported capability result.

**Implementation status (2026-07-15):** all four ports and oneCCL capability reporting are
implemented. Backend compilation and runtime validation remain on Leonardo.

## M10 — Launchers and result tooling

**Goal:** make all allreduce and MoE results discoverable and comparable.

Add complete Leonardo launcher families. Extend Benchscribe with a `case` grouping
dimension and `OK`, `NOT_IMPLEMENTED`, and `ERROR` status handling. Render unsupported
operations as `N/I`, never merge MoE routing cases, and expose MoE useful throughput and
allreduce bus bandwidth.

**Done when:** Markdown and CSV keep every message size and routing case separate,
unsupported results are excluded from speedup calculations, and parser tests cover
supported, unsupported, and malformed lines.

**Implementation status (2026-07-15):** complete locally; all seven Benchscribe tests pass.

## M11 — Documentation and matrix validation

**Goal:** finalize the six-pattern suite: `pingpong`, `halo_1d`, `allreduce`, `alltoall`,
`moe`, and `cg_step`.

Update root, backend, Leonardo, experiment-selection, roofline, and Hockney documentation.
Document timed regions, byte-count conventions, memory placement, initiation model, and the
difference between balanced dense alltoall and variable MoE exchange.

**Done when:** all six presets build, every supported benchmark/topology validates,
oneCCL capability gaps are explicit, no stale target or command remains, and Benchscribe
summarizes the full matrix without merging cases.

**Implementation status (2026-07-15):** documentation and local audits are complete;
the six accelerator presets and run matrix still require Leonardo.

## M5–M11 ordering

- M5 precedes M6 so the scalar reduction sources remain available as templates.
- M7 is independent of M5 and M6.
- M8 precedes M9, which precedes M10.
- M11 requires M5, M6, M7, and M10.
