# SYCL + oneCCL

SYCL examples using oneCCL collectives.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `sycl_oneccl_vector_add` | Each rank computes `c[i] = a[i] + b[i]` for a contiguous global slice. | oneCCL collectives distribute inputs and collect local results. |
| `sycl_oneccl_halo_1d` | Comm-only 1D halo exchange, periodic ring, swept halo width. | Point-to-point `ccl::send`/`ccl::recv` exchange device-buffer halos with both neighbors (see caveat). |
| `sycl_oneccl_dot_product` | Double-precision global dot product (CG inner-product). | `ccl::allreduce(sum)` reduces a device-resident scalar (SYCL reduction) across ranks. |
| `sycl_oneccl_pingpong` | Two-endpoint one-way latency/bandwidth, internal size sweep. | `ccl::send`/`ccl::recv` round-trip device buffers between 2 ranks (see caveat). |
| `sycl_oneccl_halo_2d` | 2D 5-point Jacobi stencil, column-slab decomposition. | `ccl::send`/`ccl::recv` exchange packed (strided) halo columns with neighbors (see caveat). |
| `sycl_oneccl_alltoall` | All-to-all personalized exchange (bisection bandwidth). | `ccl::alltoall` collective (see caveat). |
| `sycl_oneccl_cg_step` | CG iteration skeleton (SpMV halo + two reductions). | `ccl::send`/`ccl::recv` halo + two `ccl::allreduce` (see caveat). |

> **Caveat:** `sycl_oneccl_halo_1d`, `sycl_oneccl_pingpong`, `sycl_oneccl_halo_2d`,
> `sycl_oneccl_alltoall`, and `sycl_oneccl_cg_step` need oneCCL primitives the NCCL backend
> may not implement. The UNISA
> NCCL-enabled fork does not implement every primitive (e.g. `broadcast`); if `ccl::send`/
> `ccl::recv` or `ccl::alltoall` are unimplemented these binaries report a backend error
> rather than results.

## Run

```bash
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_vector_add 1048576
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_halo_1d 1048576 100 20
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_dot_product 1048576 100 20
mpirun -np 2 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_pingpong 4194304 100 20
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_halo_2d 4096 50 10
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_alltoall 65536 100 20
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_cg_step 512 50 10
```
