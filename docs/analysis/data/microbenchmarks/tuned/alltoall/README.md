# alltoall — full backend matrix, UCC enabled

**These numbers are a merge of two measured sets, not a single sweep.** Nothing
was re-run on the cluster to produce them.

| Rows | Configuration |
| --- | --- |
| `cuda_nccl`, `cuda_nvshmem`, `oshmpi`, `sycl_oneccl`, `sycl_oneccl_oshmpi` | unaffected by `OMPI_MCA_coll_ucc_enable` |
| `cuda_mpi`, `sycl_mpi` | `OMPI_MCA_coll_ucc_enable=1` |

The two MPI backends were **replaced**, not added: 238 of 833 points and 14 of
49 fits, a one-for-one swap at identical (benchmark, case, topology, backend, n)
coordinates, verified before writing. The result is the matrix as the harness now
behaves, since `alltoall`'s shim defaults `OMPI_MCA_coll_ucc_enable` to `1`.

The UCC-off arm the swap replaced is no longer in this tree. Recover it, if the
comparison is needed again, by re-running with the variable forced off:

```bash
GPU_BENCH_RESULTS_ROOT=results-ucc-off OMPI_MCA_coll_ucc_enable=0 \
  cluster/harness/launch.sh --all alltoall
```

## Why the speedup figure moved

`cuda_mpi` is the baseline every other backend is plotted against, and enabling
UCC changed it by up to 34x at small messages. Against the UCC-off figures these
replaced:

| Topology | Backend | vs `cuda_mpi` before | after |
| --- | --- | ---: | ---: |
| 4n4g | `cuda_nccl` | 6.1x faster | 0.35x |
| 4n4g | `cuda_nvshmem` | 6.2x faster | 0.35x |
| 8n4g | `cuda_nccl` | 9.3x faster | 0.37x |
| 8n4g | `oshmpi` | 4.7x faster | 0.19x |

(4-byte payload per peer.) The apparent 4.5x-9.3x advantage of NCCL, NVSHMEM and
oneCCL over MPI at small messages on 16+ ranks was an artifact of MPI running
without UCC. With UCC on, MPI is 2.7x-5x faster than all of them there.

## Caveats

- The two MPI backends carry **one job per cell**; the other five carry whatever
  the original sweep used. The `dist` figure's across-job spread is therefore not
  comparable between the two groups.
- Superseded by a real full-matrix sweep on the new default, when one is run:
  `GPU_BENCH_RESULTS_ROOT=results-v2 cluster/harness/launch.sh --all alltoall`.

## Regenerating

```bash
uv run --project tools/plot gpu-bench-plot \
  --points docs/analysis/data/microbenchmarks/tuned/alltoall/points.json \
  --fit    docs/analysis/data/microbenchmarks/tuned/alltoall/fit.json \
  --benchmark alltoall --figure all \
  --outdir docs/analysis/data/microbenchmarks/tuned/alltoall/figures
```
