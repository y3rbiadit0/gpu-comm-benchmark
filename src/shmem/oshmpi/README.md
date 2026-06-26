# OSHMPI

CUDA examples using OSHMPI/OpenSHMEM CUDA memory spaces.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `oshmpi_vector_add` | Each PE computes `c[i] = a[i] + b[i]` for a contiguous global slice. | `shmem_putmem` distributes inputs and collects local results. |
| `oshmpi_halo_1d` | One-step 1D stencil over contiguous PE-owned segments. | One-sided `shmem_putmem` writes boundary values into neighbor ghost cells. |
| `oshmpi_dot_product` | Double-precision global dot product (CG inner-product). | `shmem_double_sum_to_all` reduces a host-resident scalar (local dot computed on GPU) across PEs. |
| `oshmpi_pingpong` | Two-endpoint one-way latency/bandwidth, internal size sweep. | One-sided `shmem_putmem` on device symmetric memory + `shmem_barrier_all` handshake between 2 PEs (barrier sync avoids an inter-node passive-progress deadlock; latency includes barrier overhead). |
| `oshmpi_halo_2d` | 2D 5-point Jacobi stencil, column-slab decomposition. | One-sided `shmem_putmem` + `barrier_all` write packed (strided) halo columns into symmetric neighbor buffers. |
| `oshmpi_alltoall` | All-to-all personalized exchange (bisection bandwidth). | One-sided `shmem_putmem` loop to every PE + `barrier_all` (no native device alltoall assumed). |
| `oshmpi_cg_step` | CG iteration skeleton (SpMV halo + two reductions). | `shmem_putmem`+barrier halo + two `shmem_double_sum_to_all` (host-resident scalars). |

## Run

```bash
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_vector_add 1048576
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_halo_1d 1048576
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_dot_product 1048576 100 20
oshrun -np 2 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_pingpong 4194304 100 20
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_halo_2d 4096 50 10
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_alltoall 65536 100 20
oshrun -np 4 ./build/leonardo-oshmpi/src/shmem/oshmpi/oshmpi_cg_step 512 50 10
```
