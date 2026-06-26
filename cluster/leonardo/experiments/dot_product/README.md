# Leonardo Dot Product Experiments

These jobs benchmark a distributed double-precision dot product (the CG inner-product
motif) on Leonardo. Each rank reduces its local chunk to a scalar once, then the timed
loop measures only the cross-rank scalar **allreduce** — a latency-bound collective.
Build setup is documented in [`cluster/leonardo/README.md`](../../README.md).

## Topologies

Each backend has five fixed-topology launchers so transport effects can be isolated:

| Script | Nodes | GPUs/node | Exercises |
| --- | ---: | ---: | --- |
| `1n1g.sh` | 1 | 1 | single-GPU baseline (no inter-GPU traffic) |
| `1n2g.sh` | 1 | 2 | intra-node, NVLink |
| `1n4g.sh` | 1 | 4 | intra-node, NVLink |
| `2n1g.sh` | 2 | 1 | pure inter-node, InfiniBand |
| `2n4g.sh` | 2 | 4 | mixed intra- + inter-node |

## Submit

```bash
sbatch cluster/leonardo/experiments/dot_product/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/dot_product/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/dot_product/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/dot_product/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/dot_product/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/dot_product/sycl_oneccl/1n4g.sh
```

The intra- vs inter-node contrast is the most informative pair:

```bash
sbatch cluster/leonardo/experiments/dot_product/cuda_mpi/1n2g.sh   # NVLink
sbatch cluster/leonardo/experiments/dot_product/cuda_mpi/2n1g.sh   # InfiniBand
```

## Overrides

```bash
CP_N=1048576        # local-reduction length (does not change the allreduce volume)
CP_ITERS=200        # timed iterations
CP_WARMUP=50        # untimed warmup iterations
CP_NTRIALS=5        # job-level repeats (separate stdout files)
```

Example:

```bash
CP_ITERS=500 CP_WARMUP=100 CP_NTRIALS=5 sbatch cluster/leonardo/experiments/dot_product/cuda_nccl/2n1g.sh
```

The binaries accept `<global_size> [iterations] [warmup]`; the driver passes
`CP_ITERS`/`CP_WARMUP` through `CP_EXTRA_ARGS`.

## Output

Each run prints one standardized line per benchmark (see the root README for the schema),
for example:

```text
cuda_mpi_dot_product n=1048576 ranks=4 bytes=8 iters=100 warmup=20 time_per_iter_s=... usec=... min_usec=... max_usec=... gbytes_per_s=... validation=PASS
```

Outputs are written to:

```text
results/<result-name>/dot_product/<job-name>-<job-id>-<trial>-stdout.txt
results/<result-name>/dot_product/<job-name>-<job-id>-<trial>-stderr.txt
```

## Notes

- This is a **latency** benchmark: the reduced payload is a single scalar regardless of
  `CP_N`, so `gbytes_per_s` is not meaningful — compare `usec` / `time_per_iter_s`.
- `cuda_mpi`, `sycl_mpi`, `cuda_nccl`, `sycl_oneccl`, and `cuda_nvshmem` reduce a
  device-resident scalar. `oshmpi` reduces a host-resident scalar on the symmetric heap,
  reflecting the host-driven SHMEM-over-MPI model.
- `oneCCL`'s missing `broadcast` does not affect this benchmark; only `allreduce(sum)` is used.
