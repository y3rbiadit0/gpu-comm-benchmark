# Benchmark Output Schema

Timed benchmarks emit one standardized `key=value` record on the root rank so a
single parser can compare every backend:

```text
<name> n=<elements> ranks=<n> bytes=<per-iteration> iters=<n> warmup=<n> \
  time_per_iter_s=<seconds> usec=<microseconds> min_usec=<microseconds> \
  max_usec=<microseconds> gbytes_per_s=<GB/s> \
  [median_usec=<microseconds> p25_usec=<microseconds> \
   p75_usec=<microseconds> stddev_usec=<microseconds>] \
  [case=<case>] [status=OK|NOT_IMPLEMENTED|ERROR] \
  validation=PASS|SKIP|FAIL
```

Additional benchmark-specific fields may be present. `case` is part of the
result grouping key; MoE, for example, emits one record per routing distribution.

## Timing Semantics

Every rank records the same number of samples. An operation costs what its
slowest participant took, so the reported series is the element-wise maximum
across ranks. `time_per_iter_s` and `usec` are the mean of that reduced series.
`min_usec`, `max_usec`, and the optional percentile and standard-deviation fields
describe the same series.

Most benchmarks record one completed operation per sample. `halo_1d` records
completed batches and amortizes each sample by `batch_iters`. Its `isolated` case
uses `batch_iters=1`; its `steady` case uses the requested iteration count.

`pingpong` is the exception to the rank reduction. It reports half of the
initiator's measured round trip, so `usec` and `gbytes_per_s` are one-way values.
The peer's timing window is different and is not combined with the initiator's.

## Status And Validation

- Numeric analysis uses only `status=OK validation=PASS` records.
- Unsupported capabilities use `status=NOT_IMPLEMENTED validation=SKIP` and do
  not report usable timing values.
- Runtime errors and validation failures are excluded from numeric summaries.
- Contradictory status/validation pairs are treated as errors.

[`tools/benchscribe`](../tools/benchscribe/README.md) owns parsing, grouping, and
aggregation of this process-level output. Its versioned JSON is the separate
contract consumed by the plotting package.
