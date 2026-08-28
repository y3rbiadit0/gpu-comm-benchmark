# Clusters

The experiment system separates benchmark definitions from machine setup:

```text
cluster/
  harness/      shared launcher, matrix, jobs, and experiment defaults
  <name>/       modules, build presets, runtime tuning, and scheduler policy
```

Set `GPU_BENCH_CLUSTER=<name>` to select a machine. The shared launcher defaults
to `leonardo`.

| Cluster | Scheduler | Accelerators | Status |
| --- | --- | --- | --- |
| [`leonardo`](leonardo/README.md) | Slurm | NVIDIA A100 | Validated |

The [harness guide](harness/README.md) documents launcher syntax, result paths,
and the interface required to add another cluster.
