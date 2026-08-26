# Leonardo All-to-All Experiments

These jobs benchmark an all-to-all personalized exchange — the heaviest collective and the
classic **bisection-bandwidth** stress test (FFT / transpose / sparse / sort motif). Each
rank holds a `ranks × GPU_BENCH_N` send buffer and sends a distinct `GPU_BENCH_N`-element block to every
rank, receiving one block from each. The per-block value encodes `(source, destination)`,
so the full permutation is validated **locally** (no gather). Build setup is in
[`cluster/leonardo/README.md`](../../README.md).

## Topologies

| Script | Nodes | GPUs/node | Exercises |
| --- | ---: | ---: | --- |
| `1n1g.sh` | 1 | 1 | single-GPU baseline (self copy only) |
| `1n2g.sh` | 1 | 2 | intra-node, NVLink |
| `1n4g.sh` | 1 | 4 | intra-node, NVLink |
| `2n1g.sh` | 2 | 1 | pure inter-node, InfiniBand |
| `2n4g.sh` | 2 | 4 | mixed intra- + inter-node (full bisection) |

## Submit

```bash
tools/launch.sh alltoall cuda_mpi 1n4g
tools/launch.sh alltoall cuda_nccl 1n4g
tools/launch.sh alltoall cuda_nvshmem 1n4g
tools/launch.sh alltoall oshmpi 1n4g
tools/launch.sh alltoall sycl_mpi 1n4g
tools/launch.sh alltoall sycl_oneccl 1n4g
```

## Overrides

```bash
GPU_BENCH_N=262144         # elements exchanged with EACH peer (buffers are ranks*GPU_BENCH_N)
GPU_BENCH_ITERS=200        # timed iterations
GPU_BENCH_WARMUP=50        # untimed warmup iterations
GPU_BENCH_NTRIALS=5        # job-level repeats
```

Example:

```bash
GPU_BENCH_N=262144 GPU_BENCH_ITERS=200 tools/launch.sh alltoall cuda_nccl 2n4g
```

The binaries accept `<count_per_peer> [iterations] [warmup]`.

## Output

One standardized line per run (see the root README for the schema), e.g.:

```text
cuda_mpi_alltoall n=65536 ranks=4 bytes=1048576 iters=100 warmup=20 time_per_iter_s=... usec=... gbytes_per_s=... validation=PASS
```

`n` is the per-peer element count; `bytes` is the per-rank send volume
(`ranks * count * sizeof(float)`). Compare `gbytes_per_s` for throughput and
`usec`/`time_per_iter_s` for the per-iteration cost.

## Notes

- `cuda_mpi` / `sycl_mpi` use CUDA-aware `MPI_Alltoall`; `cuda_nvshmem` uses the native
  `nvshmem_float_alltoall` team collective; `oshmpi` performs a one-sided `shmem_putmem`
  loop with a `shmem_barrier_all` completion (no native device alltoall assumed).
- **NCCL has no native all-to-all** — `cuda_nccl` emulates it with a grouped
  `ncclSend`/`ncclRecv` to every peer. That asymmetry is itself a result worth reporting.
- **oneCCL caveat:** `sycl_oneccl_alltoall` uses `ccl::alltoall`. If the UNISA NCCL-enabled
  fork does not implement it (as with `broadcast`), this job reports a backend error — a
  documented gap, not a benchmark bug.
