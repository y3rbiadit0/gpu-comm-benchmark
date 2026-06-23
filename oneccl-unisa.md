# oneCCL UNISA Notes

These notes document the Leonardo setup used for the UNISA NCCL-enabled oneCCL fork in this playground.

## Install Layout

Expected oneCCL install prefix:

```bash
$HOME/opt/oneccl-nccl-leonardo
```

The build/runtime scripts refer to this path through:

```bash
ONECCL_NCCL_ROOT=$HOME/opt/oneccl-nccl-leonardo
```

## Bundled Intel MPI

The NCCL-enabled oneCCL fork is used with its bundled Intel MPI, not Leonardo OpenMPI, for `sycl_oneccl`.

The `sycl_oneccl_vector_add` executable calls MPI directly for setup and validation, so its linked MPI and oneCCL's dynamically loaded MPI must match. Mixing an OpenMPI-linked executable with oneCCL loading Intel MPI fails during transport initialization.

The Leonardo `leonardo-sycl-oneccl` preset enables bundled MPI mode and uses:

```text
$ONECCL_NCCL_ROOT/opt/mpi/include
$ONECCL_NCCL_ROOT/opt/mpi/lib/release
$ONECCL_NCCL_ROOT/opt/mpi/lib
```

At runtime, `cluster/leonardo/runtime/oneccl-nccl.sh` discovers bundled `libmpi.so*` under the oneCCL install and exports:

```bash
CCL_MPI_LIBRARY_PATH=/path/to/bundled/libmpi.so
```

The oneCCL vector-add jobs use `mpirun` rather than direct `srun` so the launcher also comes from the bundled MPI stack when available.

## Multi-Node Intel MPI Startup

For two or more Leonardo nodes, bundled Intel MPI needs OFI for startup. The validated setup uses Intel MPI's bundled libfabric and forces the TCP provider:

```bash
export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_FABRICS=shm:ofi
export I_MPI_DEBUG=0
export I_MPI_OFI_PROVIDER=tcp
export FI_PROVIDER=tcp
export FI_PROVIDER_PATH=$ONECCL_NCCL_ROOT/opt/mpi/libfabric/lib/prov-tcp-only
export FI_LOG_LEVEL=error
export LD_LIBRARY_PATH=$ONECCL_NCCL_ROOT/opt/mpi/libfabric/lib:${LD_LIBRARY_PATH:-}
```

`cluster/leonardo/runtime/oneccl-nccl.sh` creates the `prov-tcp-only` directory when needed by symlinking Intel MPI's bundled TCP provider:

```bash
ln -sf $ONECCL_NCCL_ROOT/opt/mpi/libfabric/lib/prov/libtcp-fi.so \
  $ONECCL_NCCL_ROOT/opt/mpi/libfabric/lib/prov-tcp-only/libtcp-fi.so
```

Without these settings, 2-node jobs can fail before oneCCL starts, inside `MPI_Init`:

```text
MPIDI_OFI_mpi_init_hook
Fatal error in internal_Init: Other MPI error
```

MPI is only the launch/bootstrap layer in this setup. Once oneCCL constructs the NCCL communicator, GPU collective payloads use NCCL.

## NCCL Backend Collective Coverage

The oneCCL NCCL backend used here does not implement `broadcast`:

```text
oneCCL: nccl_comm.cpp:355 broadcast_impl: EXCEPTION: broadcast is not implemented for NCCL backend yet
```

For `sycl_oneccl_vector_add`, rank-0 input distribution therefore uses NCCL-supported `allreduce(sum)` as a broadcast substitute:

1. Every rank initializes the full input buffers to zero.
2. Rank 0 writes the real input arrays.
3. `allreduce(sum)` over the full input buffer propagates rank 0's data because all other ranks contribute zeros.
4. The local partition is computed on each rank.
5. A final `allreduce(sum)` collects the sparse per-rank output into the full result.

This keeps the example on the oneCCL NCCL backend without relying on unsupported collectives.

## Rebuild And Smoke Test

Use a clean configure after changing MPI linkage mode:

```bash
rm -rf build/leonardo-sycl-oneccl
source cluster/leonardo/environment.sh sycl
cmake --preset leonardo-sycl-oneccl
cmake --build --preset leonardo-sycl-oneccl
```

Check linkage:

```bash
source cluster/leonardo/runtime/oneccl-nccl.sh
ldd build/leonardo-sycl-oneccl/src/sycl_oneccl/sycl_oneccl_vector_add \
  | grep -E 'libmpi|libccl|libsycl|libnccl|libstdc'
```

Run the uneven-size smoke test:

```bash
CP_N=17 CP_NTRIALS=1 sbatch cluster/leonardo/experiments/vector_add/sycl_oneccl/1n4g.sh
```

Expected result:

```text
sycl_oneccl_vector_add n=17 ranks=4 ... validation=PASS
```
