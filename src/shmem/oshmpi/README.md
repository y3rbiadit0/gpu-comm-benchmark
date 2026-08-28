# OSHMPI

This backend uses OSHMPI's OpenSHMEM API and CUDA memory-space extension for
symmetric GPU buffers.

## Implemented operations

| Benchmark | OSHMPI operation |
| --- | --- |
| `pingpong` | One-sided puts, device completion, and barrier handshake |
| `halo_1d` | Neighbor puts, quiet, device completion, and barrier |
| `allreduce` | Device or explicitly staged float sum |
| `alltoall` | Per-peer puts and barrier |
| `cg_step` | Put-based halo and two double reductions |
| `moe` | Variable-byte puts with quiet and barrier completion |

The barrier-based neighbor paths avoid passive-target progress deadlocks on the
validated inter-node transport. Barrier and device-completion costs remain in
the measured operation.

## Build

Leonardo's bootstrap installs OSHMPI, and `leonardo-oshmpi` builds the backend:

```bash
./cluster/leonardo/bootstrap.sh oneccl-oshmpi
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-oshmpi
cmake --build --preset leonardo-oshmpi
```

## CUDA memory space

`oshmpi_space.{h,c}` wraps the extension lifecycle used by every benchmark:

```text
shmemx_space_create
  -> shmemx_space_attach
    -> shmemx_space_malloc
      -> communication
    -> shmem_free
  -> shmemx_space_detach
-> shmemx_space_destroy
```

Buffers are released with `shmem_free` before their CUDA memory space is detached
and destroyed.

For benchmark semantics, see the
[benchmark contracts](../../../docs/README.md#benchmark-contracts). For
arguments, defaults, topologies, and launch examples, see the
[experiment operations](../../../cluster/harness/README.md#experiment-operations).
