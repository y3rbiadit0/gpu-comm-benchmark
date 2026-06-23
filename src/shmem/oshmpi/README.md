# OSHMPI

CUDA examples using OSHMPI/OpenSHMEM CUDA memory spaces.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `oshmpi_vector_add` | Each PE computes `c[i] = a[i] + b[i]` for a contiguous global slice. | `shmem_putmem` distributes inputs and collects local results. |
| `oshmpi_halo_1d` | One-step 1D stencil over contiguous PE-owned segments. | One-sided `shmem_putmem` writes boundary values into neighbor ghost cells. |

## Run

```bash
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_vector_add 1048576
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_halo_1d 1048576
```
