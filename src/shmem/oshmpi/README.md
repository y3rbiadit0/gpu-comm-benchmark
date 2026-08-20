# OSHMPI

CUDA examples using OSHMPI/OpenSHMEM CUDA memory spaces.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `oshmpi_halo_1d` | Comm-only 1D halo exchange, periodic ring, swept halo width. | One-sided `shmem_putmem` writes halos into symmetric neighbor buffers; `shmem_quiet` + `shmem_barrier_all` handshake (point-to-point `wait_until` can deadlock inter-node when passive RMA needs target-side progress, so the timed loop includes barrier overhead). |
| `oshmpi_pingpong` | Two-endpoint one-way latency/bandwidth, internal size sweep. | One-sided `shmem_putmem` on device symmetric memory + `shmem_barrier_all` handshake between 2 PEs (barrier sync avoids an inter-node passive-progress deadlock; latency includes barrier overhead). |
| `oshmpi_allreduce` | Float32 sum allreduce latency/bandwidth, internal size sweep. | `shmem_float_sum_to_all` over host-symmetric buffers. |
| `oshmpi_alltoall` | All-to-all personalized exchange (bisection bandwidth). | One-sided `shmem_putmem` loop to every PE + `barrier_all` (no native device alltoall assumed). |
| `oshmpi_cg_step` | CG iteration skeleton (SpMV halo + two reductions). | `shmem_putmem`+barrier halo + two `shmem_double_sum_to_all` (host-resident scalars). |
| `oshmpi_moe` | Top-1 MoE dispatch + combine with variable expert loads. | Variable-byte `shmem_putmem` loops over CUDA symmetric memory, with `quiet` + `barrier_all` after dispatch and inverse combine. |

## Run

```bash
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_halo_1d 1048576 100 20
oshrun -np 2 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_pingpong 4194304 100 20
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_allreduce 4194304 100 20
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_alltoall 65536 100 20
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_cg_step 512 50 10
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_moe 16384 256 100 20 uniform,locality80,hotspot80
```

## CUDA memory space lifecycle

`oshmpi_space.{h,c}` wraps the OSHMPI extension behind three calls so the
benchmark sources stay readable. The order matters, and matches OSHMPI's own
CUDA-space test:

```
shmemx_space_create(config with memkind = SHMEMX_MEM_CUDA)
  -> shmemx_space_attach
    -> shmemx_space_malloc  (device symmetric buffers)
      -> shmem_putmem / shmem_getmem on those buffers
    -> shmem_free           (there is no shmemx_space_free)
  -> shmemx_space_detach
-> shmemx_space_destroy
```

Buffers taken from a space are released with the ordinary `shmem_free`, before
the space they came from is detached and destroyed.
