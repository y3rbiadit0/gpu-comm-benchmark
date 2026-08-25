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
tools/sbatch.sh cluster/leonardo/experiments/cg_step/cuda_mpi/1n4g.sh
tools/sbatch.sh cluster/leonardo/experiments/cg_step/cuda_nccl/1n4g.sh
tools/sbatch.sh cluster/leonardo/experiments/cg_step/cuda_nvshmem/1n4g.sh
tools/sbatch.sh cluster/leonardo/experiments/cg_step/oshmpi/1n4g.sh
tools/sbatch.sh cluster/leonardo/experiments/cg_step/sycl_mpi/1n4g.sh
tools/sbatch.sh cluster/leonardo/experiments/cg_step/sycl_oneccl/1n4g.sh
```

### `sycl_oneccl_oshmpi`

`cg_step` needs both point-to-point (the column-slab halo) and collectives (the
two reductions), which makes it the sharpest test of the oneCCL OSHMPI backend.
The NCCL-backed fork has a **documented intra-node point-to-point failure** —
`docs/unsupported-operations.md` records `halo_1d` grouped `ccl::send`/`ccl::recv`
stalling on `1n2g`, `1n4g` and `2n4g` — and flags `cg_step` as likely to share
it. The OSHMPI backend routes point-to-point through its own slot mechanism
(`CCL_OSHMPI_PT2PT_SLOT_SIZE`), so it may work where the NCCL fork does not.

**Result (2026-08-21): the OSHMPI backend hangs; the NCCL backend works.**
`sycl_oneccl_oshmpi` on `1n2g` ran the full 2-minute timeout at 0% CPU and was
killed at oneCCL initialization. `sycl_oneccl` — the same benchmark, same
topology, NCCL backend — completed in 2.0 s with `validation=PASS`. So this is
an OSHMPI-backend fault, not a oneCCL grouped-point-to-point limitation.
Recorded in [`docs/unsupported-operations.md`](../../../../docs/unsupported-operations.md).

Do not read the working `1n2g` result as clearance for `sycl_oneccl` generally.
On `1n2g` cg_step's non-periodic line gives each rank one neighbour, so the
group holds two operations. `halo_1d`'s periodic ring gives it four, and that is
the case that stalls. On `1n4g` cg_step's interior ranks have two neighbours and
post four grouped operations — the same shape as the stalling `halo_1d` run:

```bash
# the discriminating test
GPU_BENCH_NTRIALS=1 tools/sbatch.sh \
  cluster/leonardo/experiments/cg_step/sycl_oneccl/1n4g.sh

# retest the OSHMPI backend against a fixed build
GPU_BENCH_NTRIALS=1 tools/sbatch.sh \
  cluster/leonardo/experiments/cg_step/sycl_oneccl_oshmpi/1n2g.sh
```

## Overrides

```bash
GPU_BENCH_N=8192           # grid side length (grid is GPU_BENCH_N x GPU_BENCH_N)
GPU_BENCH_ITERS=100        # timed CG-step iterations
GPU_BENCH_WARMUP=20        # untimed warmup iterations
GPU_BENCH_NTRIALS=5        # job-level repeats
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
