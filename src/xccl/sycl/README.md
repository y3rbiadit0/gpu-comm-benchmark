# SYCL + oneCCL

SYCL examples using oneCCL collectives.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `sycl_oneccl_halo_1d` | Comm-only 1D halo exchange, periodic ring, swept halo width. | Grouped point-to-point `ccl::send`/`ccl::recv` exchange device-buffer halos with both neighbors (see caveat). |
| `sycl_oneccl_pingpong` | Two-endpoint one-way latency/bandwidth, internal size sweep. | `ccl::send`/`ccl::recv` round-trip device buffers between 2 ranks (see caveat). |
| `sycl_oneccl_allreduce` | Float32 sum allreduce latency/bandwidth, internal size sweep. | `ccl::allreduce(sum)` over device buffers. |
| `sycl_oneccl_alltoall` | All-to-all personalized exchange (bisection bandwidth). | `ccl::alltoall` collective (see caveat). |
| `sycl_oneccl_cg_step` | CG iteration skeleton (SpMV halo + two reductions). | `ccl::send`/`ccl::recv` halo + two `ccl::allreduce` (see caveat). |
| `sycl_oneccl_moe` | Top-1 MoE dispatch + combine with variable expert loads. | Two sets of per-peer variable-count `ccl::send`/`ccl::recv` operations: dispatch followed by inverse combine (see caveat). |

> **Caveat:** `sycl_oneccl_halo_1d`, `sycl_oneccl_pingpong`,
> `sycl_oneccl_alltoall`, `sycl_oneccl_cg_step`, and `sycl_oneccl_moe` need oneCCL primitives the NCCL backend
> may not implement. The UNISA
> NCCL-enabled fork does not implement every primitive (e.g. `broadcast`); if `ccl::send`/
> `ccl::recv` or `ccl::alltoall` are unimplemented, the non-MoE binaries report a backend
> error rather than results. MoE detects unsupported point-to-point operations collectively,
> emits `status=NOT_IMPLEMENTED reason=point_to_point validation=SKIP` for the affected and
> remaining routing cases, and exits successfully so unsupported capability is not reported
> as a failed benchmark.
>
> On the validated Leonardo fork, the grouped `halo_1d` ring stalls whenever a
> node hosts more than one participating rank. The `1n2g`, `1n4g`, and `2n4g`
> jobs therefore report `NOT_IMPLEMENTED`; `2n1g` remains supported. See the
> [unsupported operations tracker](../../../docs/unsupported-operations.md).

## Run

```bash
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_halo_1d 1048576 100 20
mpirun -np 2 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_pingpong 4194304 100 20
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_allreduce 4194304 100 20
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_alltoall 65536 100 20
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_cg_step 512 50 10
mpirun -np 4 ./build/leonardo-sycl-oneccl/src/xccl/sycl/sycl_oneccl_moe 16384 256 100 20 uniform,locality80,hotspot80
```
