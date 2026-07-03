# Communication Playground

Small distributed GPU communication examples organized by communication model.

## Layout

```text
src/
  mpi/      # CUDA MPI and SYCL MPI
  xccl/     # CUDA NCCL and SYCL oneCCL
  shmem/    # NVSHMEM and OSHMPI
```

## Benchmarks

| Benchmark | Purpose |
| --- | --- |
| `vector_add` | Basic distributed execution |
| `halo_1d` | Neighbor communication and one-sided models ([guide](docs/halo_1d.md)) |
| `dot_product` | Global reduction / allreduce latency (CG inner-product motif) |
| `pingpong` | Point-to-point one-way latency and bandwidth (message-size sweep) |
| `alltoall` | All-to-all personalized exchange (bisection bandwidth) |
| `cg_step` | Conjugate-gradient iteration skeleton (SpMV halo + two reductions) |

## Implementations

| Model | Backend | Details |
| --- | --- |
| MPI | CUDA | [`src/mpi/cuda`](src/mpi/cuda/README.md) |
| MPI | SYCL | [`src/mpi/sycl`](src/mpi/sycl/README.md) |
| XCCL | CUDA/NCCL | [`src/xccl/cuda`](src/xccl/cuda/README.md) |
| XCCL | SYCL/oneCCL | [`src/xccl/sycl`](src/xccl/sycl/README.md) |
| SHMEM | NVSHMEM | [`src/shmem/nvshmem`](src/shmem/nvshmem/README.md) |
| SHMEM | OSHMPI | [`src/shmem/oshmpi`](src/shmem/oshmpi/README.md) |

## Build

Build one preset directly:

```bash
make configure PRESET=cuda-mpi
make build PRESET=cuda-mpi
```

Build all Leonardo presets on Leonardo:

```bash
make leonardo
```

Use `make leonardo-cuda` or `make leonardo-sycl` to build only one stack. The Makefile is only a thin wrapper over `CMakePresets.json`.

## Run Examples

```bash
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_vector_add 1048576
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_halo_1d 1048576 100 20
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_dot_product 1048576 100 20
mpirun -np 2 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_pingpong 4194304 100 20
mpirun -np 2 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_pingpong 4194304 100 20 1,8,64,1024
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_alltoall 65536 100 20
mpirun -np 4 ./build/cuda-mpi/src/mpi/cuda/cuda_mpi_cg_step 512 50 10
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_vector_add 1048576
mpirun -np 4 ./build/sycl-mpi/src/mpi/sycl/sycl_mpi_halo_1d 1048576 100 20
```

Each binary accepts the global problem size as the first argument. Iterative halo variants also accept an iteration count as the second argument. `dot_product` binaries accept `<global_size> [iterations] [warmup]`. `pingpong` binaries require exactly 2 ranks, accept `<max_elements> [iterations] [warmup] [message_sizes]`, and sweep message sizes internally (one report line per size). If `[message_sizes]` is omitted, pingpong uses powers of two from 1 to `<max_elements>`; otherwise pass comma-separated element counts such as `1,8,64,1024`. `alltoall` binaries accept `<count_per_peer> [iterations] [warmup]`; each rank exchanges `count_per_peer` elements with every rank (send/recv buffers are `ranks × count_per_peer`). `cg_step` binaries accept `<side> [iterations] [warmup]` and run one CG-iteration communication skeleton per step: an SpMV (column-slab halo exchange + 5-point stencil) followed by two global reductions — combining the column-slab halo-exchange and `dot_product` patterns.

## Benchmark Output Schema

`dot_product` (and future benchmarks built on `include/timing.hpp` + `include/report.hpp`) emit a single standardized `key=value` line on the root rank so one parser can compare every backend:

```text
<name> n=<elements> ranks=<n> bytes=<per-iter> iters=<n> warmup=<n> \
  time_per_iter_s=<s> usec=<us> min_usec=<us> max_usec=<us> gbytes_per_s=<gb/s> validation=PASS|FAIL
```

`time_per_iter_s`/`usec` is the slowest-rank average over the timed loop (after warmup);
`min_usec`/`max_usec` bound the per-iteration distribution. For latency benchmarks such as
`dot_product` the payload is a scalar, so compare `usec` rather than `gbytes_per_s`.

`pingpong` reports **one-way** figures (half the measured round trip): `usec` is one-way
latency and `gbytes_per_s` is one-way bandwidth, with one line per swept message size.

## Summarizing Results

`tools/benchscribe` parses the `results/` tree, aggregates across trials, and
prints a comparison table where every backend is normalized to the `cuda_mpi` baseline:

```bash
python3 tools/benchscribe                       # Markdown, all benchmarks
python3 tools/benchscribe --benchmark dot_product
python3 tools/benchscribe --format csv > summary.csv
```

See [`tools/README.md`](tools/README.md) for details.

## Leonardo

Use the Slurm scripts for validation runs; they request the GPU partition and resources correctly.

```bash
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/vector_add/cuda_mpi/1n1g.sh
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/cuda_mpi/1n1g.sh
```

Leonardo setup, presets, and experiment notes live under [`cluster/leonardo`](cluster/leonardo/README.md).
