# SYCL + oneCCL

This backend uses oneCCL with SYCL USM device buffers. The same sources are built
against NCCL and OSHMPI oneCCL transports on Leonardo.

## Implemented operations

| Benchmark | oneCCL operation |
| --- | --- |
| `pingpong` | Point-to-point send and receive |
| `halo_1d` | Grouped neighbor sends and receives |
| `allreduce` | `ccl::allreduce` with sum |
| `alltoall` | `ccl::alltoall` |
| `cg_step` | Grouped halo exchange and two allreduces |
| `moe` | Variable-count point-to-point dispatch and combine |

oneCCL capabilities depend on the configured transport. MoE collectively
recognizes an unavailable point-to-point API and emits
`status=NOT_IMPLEMENTED validation=SKIP`; unexpected failures in other binaries
remain errors.

## Build

Leonardo provides two presets and runtime backend names:

| Transport | Preset | Harness backend |
| --- | --- | --- |
| NCCL | `leonardo-sycl-oneccl` | `sycl_oneccl` |
| OSHMPI | `leonardo-sycl-oneccl-oshmpi` | `sycl_oneccl_oshmpi` |

```bash
source cluster/leonardo/environment.sh sycl
cmake --preset leonardo-sycl-oneccl
cmake --build --preset leonardo-sycl-oneccl
cmake --preset leonardo-sycl-oneccl-oshmpi
cmake --build --preset leonardo-sycl-oneccl-oshmpi
```

The [support matrix](../../../docs/reference/support-matrix.md) lists which
benchmarks are declared for each transport.

For benchmark semantics, see the
[benchmark contracts](../../../docs/README.md#benchmark-contracts). For
arguments, defaults, topologies, and launch examples, see the
[experiment operations](../../../cluster/harness/README.md#experiment-operations).
