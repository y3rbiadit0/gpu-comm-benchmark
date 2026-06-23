# SYCL + oneCCL

SYCL examples using oneCCL collectives.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `sycl_oneccl_vector_add` | Each rank computes `c[i] = a[i] + b[i]` for a contiguous global slice. | oneCCL collectives distribute inputs and collect local results. |
| `sycl_oneccl_halo_1d` | One-step 1D stencil over contiguous rank-owned segments. | Collective emulation with `allreduce(sum)` because oneCCL/NCCL has no natural neighbor halo primitive. |

## Run

```bash
mpirun -np 4 ./build/leonardo-sycl-oneccl/implementations/sycl_oneccl/sycl_oneccl_vector_add 1048576
mpirun -np 4 ./build/leonardo-sycl-oneccl/implementations/sycl_oneccl/sycl_oneccl_halo_1d 1048576
```
