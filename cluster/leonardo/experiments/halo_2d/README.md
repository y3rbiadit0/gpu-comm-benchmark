# Leonardo Halo 2D Experiments

These jobs benchmark a 2D 5-point Jacobi stencil halo exchange on Leonardo. The square
`CP_N × CP_N` grid is split across ranks with a **column-slab decomposition**: each rank
owns a contiguous range of columns (all rows). The halo is therefore the first/last
interior **column**, which is *strided* in row-major storage and is packed into a
contiguous buffer before exchange and unpacked afterwards — the property that distinguishes
this from `halo_1d`. Each timed iteration performs one halo exchange plus one stencil step.
Validation is **local** (closed-form reference `f(i,j)=i+j`), so no gather is needed.
Build setup is in [`cluster/leonardo/README.md`](../../README.md).

## Topologies

| Script | Nodes | GPUs/node | Exercises |
| --- | ---: | ---: | --- |
| `1n1g.sh` | 1 | 1 | single-GPU baseline (no halo traffic) |
| `1n2g.sh` | 1 | 2 | intra-node, NVLink |
| `1n4g.sh` | 1 | 4 | intra-node, NVLink |
| `2n1g.sh` | 2 | 1 | pure inter-node, InfiniBand |
| `2n4g.sh` | 2 | 4 | mixed intra- + inter-node |

## Submit

```bash
sbatch cluster/leonardo/experiments/halo_2d/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_2d/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/halo_2d/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/halo_2d/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_2d/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/halo_2d/sycl_oneccl/1n4g.sh
```

## Overrides

```bash
CP_N=8192           # grid side length (grid is CP_N x CP_N)
CP_ITERS=100        # timed halo+stencil steps
CP_WARMUP=20        # untimed warmup steps
CP_NTRIALS=5        # job-level repeats
```

Example:

```bash
CP_N=8192 CP_ITERS=200 sbatch cluster/leonardo/experiments/halo_2d/cuda_nvshmem/2n4g.sh
```

The binaries accept `<side> [iterations] [warmup]`.

## Output

One standardized line per run (see the root README for the schema), e.g.:

```text
cuda_mpi_halo_2d n=4096 ranks=4 bytes=32768 iters=50 warmup=10 time_per_iter_s=... usec=... gbytes_per_s=... validation=PASS
```

`n` is the grid side length; `bytes` is the nominal halo volume exchanged by an interior
rank per step (`2 * side * sizeof(float)`). Compare `usec`/`time_per_iter_s` for the
per-step cost.

## Notes

- `cuda_mpi` / `sycl_mpi` exchange packed columns with CUDA-aware `MPI_Sendrecv`;
  `cuda_nccl` uses grouped `ncclSend`/`ncclRecv`; `cuda_nvshmem` and `oshmpi` use host-driven
  one-sided `put` into symmetric column buffers with a barrier.
- **oneCCL caveat:** `sycl_oneccl_halo_2d` uses `ccl::send`/`ccl::recv` for a true halo. If
  the UNISA NCCL-enabled fork does not implement point-to-point (as with `broadcast`), this
  job reports a backend error — a documented gap, consistent with `sycl_oneccl_pingpong`.
- Single-GPU `1n1g` performs no halo communication (no neighbours); it measures the stencil
  and pack/unpack floor.
