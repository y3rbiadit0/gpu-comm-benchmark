# Leonardo CG Step Experiments

These jobs benchmark the **communication skeleton of a conjugate-gradient iteration** on a
2D 5-point Laplacian — the composite motif that mixes neighbor and collective communication
in one step:

1. **SpMV** `q = A·p` — a column-slab halo exchange of `p` (strided column, packed) followed
   by the 5-point stencil (column-slab halo exchange).
2. **Two global reductions** `dot(p,q)` and `dot(q,q)` — the dot products a CG iteration needs
   for its `alpha`/`beta` coefficients.

So each timed iteration is `1 halo exchange + 2 allreduces`, the real CG bottleneck. `p ≡ 1`,
which makes both reductions exactly checkable (`dot(p,q) = S(S-1)`) and makes `q` validate the
halo. Validation is local (no gather). Build setup is in
[`cluster/leonardo/README.md`](../../README.md).

## Topologies

| Script | Nodes | GPUs/node | Exercises |
| --- | ---: | ---: | --- |
| `1n1g.sh` | 1 | 1 | single-GPU baseline (reductions trivial, no halo) |
| `1n2g.sh` | 1 | 2 | intra-node, NVLink |
| `1n4g.sh` | 1 | 4 | intra-node, NVLink |
| `2n1g.sh` | 2 | 1 | pure inter-node, InfiniBand |
| `2n4g.sh` | 2 | 4 | mixed intra- + inter-node |

## Submit

```bash
sbatch cluster/leonardo/experiments/cg_step/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/cg_step/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/cg_step/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/cg_step/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/cg_step/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/cg_step/sycl_oneccl/1n4g.sh
```

## Overrides

```bash
CP_N=8192           # grid side length (grid is CP_N x CP_N)
CP_ITERS=100        # timed CG-step iterations
CP_WARMUP=20        # untimed warmup iterations
CP_NTRIALS=5        # job-level repeats
```

The binaries accept `<side> [iterations] [warmup]`.

## Output

One standardized line per run (see the root README for the schema), e.g.:

```text
cuda_mpi_cg_step n=4096 ranks=4 bytes=32768 iters=50 warmup=10 time_per_iter_s=... usec=... validation=PASS
```

`n` is the grid side; `bytes` is the halo volume per interior rank (the two reductions are
scalars). Compare `usec`/`time_per_iter_s` for the combined per-iteration cost — this is the
benchmark that shows how each backend handles **mixed** neighbor + collective traffic (e.g.
NVSHMEM's device-initiated model vs. host-driven collectives).

## Notes

- Each backend uses its native communication mechanisms: MPI uses CUDA-aware `Sendrecv` +
  `MPI_Allreduce`; NCCL uses grouped `ncclSend`/`ncclRecv` + `ncclAllReduce`; NVSHMEM uses
  host-driven `put`+barrier + `nvshmem_double_sum_reduce`; OSHMPI uses `putmem`+barrier +
  `shmem_double_sum_to_all`.
- The two reductions are issued **separately** (as a real CG iteration does), so the
  benchmark reflects two reduction latencies per step, not one fused reduction.
- `sycl_oneccl_cg_step` uses grouped `ccl::send`/`ccl::recv` for the halo. The
  Leonardo backend routes oneCCL groups through `group_impl` to native NCCL
  groups, and grouped point-to-point is enabled by default.
