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

The `sycl_oneccl` executables call MPI directly for setup and validation, so their linked MPI and oneCCL's dynamically loaded MPI must match. Mixing an OpenMPI-linked executable with oneCCL loading Intel MPI fails during transport initialization.

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

The oneCCL jobs use `mpirun` rather than direct `srun` so the launcher also comes from the bundled MPI stack when available.

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

The active benchmark suite does not require this unsupported collective. Any future
broadcast benchmark must either wait for backend support or report the capability gap
explicitly rather than silently changing the operation.

Point-to-point support is also required by halo, pingpong, CG halo exchange, and MoE.
The MoE benchmark treats a recognized missing `ccl::send`/`ccl::recv` implementation as a
capability result and emits `status=NOT_IMPLEMENTED reason=point_to_point validation=SKIP`.
It never falls back to MPI, leaving the missing oneCCL operation visible for contribution.

The local oneCCL source keeps `ccl_api_functions.cpp` backend-neutral and
dispatches groups through the existing `group_impl` layer. Its NCCL path uses
native `ncclGroupStart`/`ncclGroupEnd`, preserves outermost-group nesting, defers
per-operation completion, and publishes SYCL stream events only after NCCL has
enqueued the group. Grouped point-to-point is enabled by default for `halo_1d`,
multi-rank CG, and MoE. The `halo_1d` ring is validated on `1n2g`, `1n4g`,
`2n1g`, and `2n4g`.

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
ldd build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_allreduce \
  | grep -E 'libmpi|libccl|libsycl|libnccl|libstdc'
```

Run an explicit-size smoke test:

```bash
GPU_BENCH_MSG_SIZES=17 GPU_BENCH_NTRIALS=1 sbatch cluster/leonardo/experiments/allreduce/sycl_oneccl/1n4g.sh
GPU_BENCH_N=17 GPU_BENCH_NTRIALS=1 sbatch cluster/leonardo/experiments/halo_1d/sycl_oneccl/1n2g.sh
```

Expected result:

```text
sycl_oneccl_allreduce n=17 ranks=4 ... validation=PASS
sycl_oneccl_halo_1d n=17 ranks=2 ... validation=PASS
```
