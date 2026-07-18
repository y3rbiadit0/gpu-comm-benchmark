# Experiment Selection and Classification

Why these communication patterns, how they are classified, and how the suite maps onto the
industry-standard benchmarks (OSU Micro-Benchmarks 7.5.2). This is the rationale behind the
experiments under `cluster/leonardo/experiments/`.

## Scope

Six patterns make up the results set:

- `pingpong`
- `halo_1d`
- `allreduce`
- `alltoall`
- `cg_step`
- `moe`

The full experiment design is a three-axis matrix:

1. **Pattern** — the communication shape (this document).
2. **Backend** — the programming/sync model: two-sided message passing (`cuda_mpi`,
   `sycl_mpi`, `cuda_nccl`, `sycl_oneccl`), device-initiated one-sided put+signal
   (`cuda_nvshmem`), or host-initiated one-sided put+quiet with global barrier
   completion (`oshmpi`).
3. **Topology** — the transport: intra-node NVLink (`1n2g`, `1n4g`), pure inter-node
   InfiniBand (`2n1g`), and mixed (`2n4g`), with `1n1g` as the no-communication baseline.

## Classification

Two orthogonal axes: *who talks to whom* (communication scope) and *what limits it*
(the dominant term of the α–β cost model, where α is per-message latency and β is
inverse bandwidth).

### Axis 1 — Communication scope

Matches the taxonomy OSU uses for its own benchmark tree (pt2pt / neighborhood /
collective).

| Scope | Pattern | Peers per rank | Traffic per rank as P grows |
| --- | --- | ---: | --- |
| Point-to-point | `pingpong` | 1 | O(1) — independent of P |
| Neighbor exchange (sparse, fixed degree) | `halo_1d` | 2 (periodic ring) | O(1) — constant regardless of P |
| Global collective, reduction | `allreduce` | all, via tree/ring | algorithm-dependent latency and bandwidth |
| Global collective, dense personalized | `alltoall` | all, equal block to each | O(P) messages — stresses bisection bandwidth |
| Global personalized application pattern, variable/skewed | `moe` | routing-dependent subset of all ranks | O(P) possible peers; per-peer sizes and receive loads vary |
| Composite / application skeleton | `cg_step` | mixed: 2 neighbors + 2 collectives | one CG iteration |

### Axis 2 — Performance regime

| Regime | Pattern | Why |
| --- | --- | --- |
| Bandwidth-bound (β) | `alltoall` | bisection bandwidth is the whole point |
| Swept across both | `pingpong`, `halo_1d`, `allreduce` | the message-size sweep crosses from the α regime to the β regime; the crossover n½ is itself a result (see `docs/analysis/halo_1d-crossover.md`) |
| Mixed | `cg_step` | two α-bound allreduces + one β-bound halo per iteration; not predictable from either regime alone |
| Bandwidth- and imbalance-sensitive | `moe` | two variable-count global exchanges move a fixed useful payload, while routing controls locality, per-peer message sizes, and the busiest expert |

The axes are kept separate on purpose: `pingpong` is not "the latency benchmark" — it is
the point-to-point pattern measured across both regimes. Its small-message and
large-message asymptotes are the fitted α and β per transport. The other patterns then
test whether those fitted parameters, pushed through each pattern's cost model, predict
the measured numbers:

- `halo_1d` ring exchange: ≈ 2α + 2βn per iteration
- `allreduce`: algorithm-dependent tree/ring latency plus payload transfer
- `alltoall`: bisection-limited, per-rank bytes grow with P
- `cg_step`: sum of the halo and allreduce terms per CG iteration
- `moe`: dispatch + inverse combine; useful bytes are fixed by tokens and hidden width, but
  runtime follows variable peer traffic and the maximum expert load

Where the prediction holds, the α–β model explains the library; where it breaks, that is a
finding about the implementation.

## Mapping onto OSU Micro-Benchmarks 7.5.2

OSU is the de facto industry yardstick; its tree ships `mpi/pt2pt`, `mpi/collective`
(blocking / non-blocking / persistent / **neighborhood**), `mpi/one-sided`, `openshmem`,
and `xccl` (NCCL/RCCL) categories — independently validating both the pattern set and the
choice to benchmark MPI, NCCL/oneCCL, and NVSHMEM/OSHMPI as peer backends.

| Suite pattern | OSU equivalent | Notes |
| --- | --- | --- |
| `pingpong` | `osu_latency` / `osu_bw`, `osu_xccl_latency`, `osu_oshm_put` | same OSU-style size-sweep methodology |
| `allreduce` | `osu_allreduce` / `osu_xccl_allreduce` | same message-size sweep and collective sum |
| `halo_1d` | `collective/neighborhood` category (`osu_neighbor_alltoall`, …) | OSU dedicates a whole category to neighbor exchange |
| `alltoall` | `osu_alltoall` / `osu_xccl_alltoall` | classic bisection-bandwidth stress |
| `cg_step` | **none** | OSU only measures isolated primitives; a composite application skeleton is this suite's value-add |
| `moe` | **none** | variable-count, routing-skewed dispatch + combine is an application pattern rather than dense `osu_alltoall` |

Calibration: building OSU 7.5.2 with `--enable-cuda` on the target cluster and running
`osu_latency`, `osu_bw`, `osu_allreduce`, `osu_alltoall` (plus the `xccl` variants) next to
the suite's `cuda_mpi`/`cuda_nccl` numbers gives an external ground truth for the harness —
if the numbers agree within a few percent, the measurement methodology is validated
independently.

## Extension policy and known gaps

The suite covers point-to-point, reduction collective, neighbor exchange, dense personalized
collective, variable/skewed personalized application traffic, and composite communication. New
patterns are added only when they introduce a new *class* of communication; each addition
costs one source per backend + generated launchers, and `tools/benchscribe` picks it up
automatically.

The known gap is broadcast, which remains blocked until the oneCCL fork implements it.
