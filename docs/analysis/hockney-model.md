# Hockney Model Rationale

The suite uses the Hockney latency/bandwidth form as a descriptive model for
message-size sweeps:

```text
T(m) = alpha + m / B_inf
```

This choice is informed by:

> J. A. Rico-Gallego, J. C. Diaz-Martin, R. R. Manumachu, and A. L. Lastovetsky,
> "A Survey of Communication Performance Models for High-Performance
> Computing," ACM Computing Surveys 51(6), 2019.
> [doi:10.1145/3284358](https://doi.org/10.1145/3284358)

## Why It Fits The Microbenchmarks

For `pingpong` and `halo_1d`, the model separates two observable regimes:

- a small-message floor dominated by fixed launch, matching, progress, and
  completion costs; and
- a large-message region dominated by transfer bandwidth.

The suite uses the parameters to characterize measured curves and locate
crossovers, not to predict complete application runtime. More detailed LogP,
LogGP, or LogGPS parameters would require host-overhead isolation procedures
that do not translate consistently to device-initiated NVSHMEM or queued NCCL
execution.

The halo ring also measures concurrent exchange traffic. In the survey's
terminology it resembles a ring operation with transfer concurrency two. Its
bandwidth parameter therefore describes the complete exchange and is not
equivalent to ping-pong bandwidth.

## Limits

A single alpha/bandwidth pair cannot represent every feature of a modern GPU
communication curve. Eager/rendezvous transitions, NCCL protocol changes,
transport switches, and large-message cliffs create multiple regimes. For that
reason Benchscribe reports observed latency-floor and peak-bandwidth anchors,
and the analysis keeps the complete curve visible.

The model is also insufficient for predicting collectives or application
patterns. Collective algorithms depend on rank count, topology, and operation
schedule; applications add compute and overlap. `allreduce`, `alltoall`,
`cg_step`, and `moe` are measured directly rather than inferred from one
point-to-point equation.

## Statistical Interpretation

Independent jobs and within-job samples answer different questions. Within-job
spread captures iteration-level jitter on one allocation. Across-job spread
captures placement and system variation and is the relevant quantity for
reproducibility. Benchscribe preserves both in JSON, and GPU Bench Plot displays
them separately in distribution figures.

The concrete characterization and reproduction workflow is documented in the
[`halo_1d` analysis methodology](halo-1d-methodology.md).
