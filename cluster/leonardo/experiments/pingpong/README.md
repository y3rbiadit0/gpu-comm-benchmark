# Leonardo Ping-Pong Experiments

These jobs benchmark point-to-point **one-way latency and bandwidth** between two GPUs
(OSU-style ping-pong). Rank 0 sends a message to rank 1, which echoes it back; the round
trip is timed on the initiator and halved to report one-way figures. Each binary sweeps
message sizes internally, printing one standardized line per size. By default the sweep
uses powers of two from 1 element up to `CP_N`; set `CP_MSG_SIZES` to a comma-separated
list to run specific element counts. Build setup is in [`cluster/leonardo/README.md`](../../README.md).

## Topologies

Ping-pong needs exactly two endpoints, so only two launchers exist per backend — chosen to
isolate the transport:

| Script | Nodes | GPUs/node | Transport |
| --- | ---: | ---: | --- |
| `1n2g.sh` | 1 | 2 | intra-node NVLink |
| `2n1g.sh` | 2 | 1 | inter-node InfiniBand |

## Submit

```bash
sbatch cluster/leonardo/experiments/pingpong/cuda_mpi/1n2g.sh
sbatch cluster/leonardo/experiments/pingpong/cuda_mpi/2n1g.sh
sbatch cluster/leonardo/experiments/pingpong/cuda_nccl/2n1g.sh
sbatch cluster/leonardo/experiments/pingpong/cuda_nvshmem/2n1g.sh
sbatch cluster/leonardo/experiments/pingpong/oshmpi/2n1g.sh
sbatch cluster/leonardo/experiments/pingpong/sycl_mpi/2n1g.sh
sbatch cluster/leonardo/experiments/pingpong/sycl_oneccl/2n1g.sh
```

## Overrides

```bash
CP_N=16777216       # maximum message length in elements (float); sweep goes 1..CP_N by x2
CP_MSG_SIZES=1,8,64,1024,1048576  # optional explicit message lengths in elements
CP_ITERS=200        # timed round trips per size
CP_WARMUP=50        # untimed warmup round trips per size
CP_NTRIALS=5        # job-level repeats
```

Example:

```bash
CP_N=16777216 CP_ITERS=500 sbatch cluster/leonardo/experiments/pingpong/cuda_nvshmem/1n2g.sh
CP_MSG_SIZES=1,8,64,1024,1048576 sbatch cluster/leonardo/experiments/pingpong/cuda_mpi/2n1g.sh
```

## Output

One line per swept message size, e.g.:

```text
cuda_mpi_pingpong n=1024 ranks=2 bytes=4096 iters=100 warmup=20 time_per_iter_s=... usec=<one-way latency> ... gbytes_per_s=<one-way bandwidth> validation=PASS
```

`usec` is one-way latency (half the round trip); `gbytes_per_s` is one-way bandwidth.
Plot `gbytes_per_s` vs `bytes` for the bandwidth curve and read small-message `usec` for
the latency floor.

## Notes

- All backends ping-pong **device-resident** buffers. CUDA/SYCL MPI use CUDA-aware
  `Send`/`Recv`; NCCL uses `ncclSend`/`ncclRecv`; OSHMPI uses host-driven one-sided
  `shmem_putmem` with a `shmem_barrier_all` handshake (a point-to-point `wait_until` flag
  deadlocks inter-node when passive RMA needs target-side progress, so the barrier — which
  always progresses — is used instead; OSHMPI latency therefore includes barrier overhead).
  NVSHMEM is **device-initiated** (the round-trip loop runs inside
  one kernel using `nvshmemx_signal_op`/`nvshmem_signal_wait_until`, since NVSHMEM
  point-to-point synchronization has no host variant); its per-size time is therefore the
  amortized per-round-trip latency (no per-iteration min/max distribution).
- **oneCCL caveat:** `sycl_oneccl_pingpong` uses `ccl::send`/`ccl::recv`. The UNISA
  NCCL-enabled oneCCL fork does not implement every primitive (e.g. `broadcast`); if
  point-to-point is likewise unimplemented this job will report a backend error. Treat that
  as a documented backend gap, not a benchmark bug.
