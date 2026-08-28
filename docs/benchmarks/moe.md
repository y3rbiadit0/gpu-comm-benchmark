# Mixture-of-Experts Benchmark

`moe` measures a top-1 mixture-of-experts communication step with deterministic
uniform, locality-biased, and hotspot routing. It includes dispatch and inverse
combine but no expert computation.

## Operation

Every rank starts with `T` tokens containing `H` float32 features. Each token is
assigned to one expert rank, and tokens are packed into destination-ordered
blocks. Dispatch sends those variable-sized blocks to their expert ranks. The
inverse combine returns each received block to its source rank without modifying
the payload.

Planning, route generation, and packing happen before timing. The benchmark
provides three deterministic routing cases:

| Case | Distribution |
| --- | --- |
| `uniform` | Tokens are assigned by a uniform hash across ranks |
| `locality80` | Approximately 80% remain on their source rank; the rest use another rank |
| `hotspot80` | Approximately 80% target rank 0; the rest use another rank |

At one rank, all three cases are local and operationally equivalent.

## Command-line contract

```text
<tokens_per_rank> [hidden] [iterations] [warmup] [comma-separated routing cases]
```

Numeric arguments must be positive. Routing names must be selected from
`uniform`, `locality80`, and `hotspot80`.

Harness defaults, valid topologies, and environment overrides are documented in
the [experiment operations](../../cluster/harness/experiments/moe/README.md).

## Timing

Each sample includes one completed dispatch and inverse combine. Planning,
packing, allocation, initialization, the initial rank barrier, validation,
timing reduction, and reporting are outside timing.

Local samples are reduced element-wise with the maximum across ranks, and all
summary statistics are calculated from that globally reduced series:

```text
global_sample[i] = max(local_sample[rank][i])
```

## Validation

After timing, every rank validates the dispatch layout, token ownership,
source-token ordering, and deterministic payload. It then checks that combine
reproduced the packed send buffer exactly. Rank-local verdicts are combined into
one global result.

## Metrics

Useful bytes count both dispatch and combine for every token, including tokens
that remain on their source rank:

```text
bytes = 2 * T * H * sizeof(float) = 8 * T * H
gbytes_per_s = bytes / time_per_iteration
useful_gbytes_per_s = gbytes_per_s
```

The report also includes the routing case, token and hidden sizes, `top_k=1`, the
maximum tokens assigned to any expert, and:

```text
expert_imbalance = max_expert_tokens / tokens_per_rank
```

The byte count is fixed across routing cases and is useful application volume,
not exact traffic crossing physical links.

## Completion boundaries

| Harness backend | Dispatch and combine model | Completion boundary |
| --- | --- | --- |
| `cuda_mpi` | Two variable-count `MPI_Alltoallv` calls | Return of the blocking collectives |
| `sycl_mpi` | `MPI_Alltoallv` over USM buffers | Return of the blocking collectives |
| `cuda_nccl` | Two grouped variable-count exchanges | CUDA stream synchronization |
| `cuda_nvshmem` | Per-destination puts and local copies | Quiet and global barrier after each phase |
| `oshmpi` | Variable-byte puts and local copies | Quiet, device synchronization, and barrier after each phase |
| `sycl_oneccl` | Grouped variable-count point-to-point exchanges | oneCCL event completion |

Recognized missing oneCCL point-to-point support is reported as
`status=NOT_IMPLEMENTED validation=SKIP`; unexpected or inconsistent failures
remain benchmark errors. The oneCCL/OSHMPI transport is not declared for MoE.

## Related guides

- [Experiment operations](../../cluster/harness/experiments/moe/README.md):
  defaults, supported topologies, overrides, and launch examples
- [Backend implementations](../../src/README.md): library operations and build
  instructions
- [Support matrix](../reference/support-matrix.md): declared backend coverage
- [Output schema](../reference/output-schema.md): common fields and timing
  reduction rules
