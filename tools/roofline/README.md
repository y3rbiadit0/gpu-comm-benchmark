# Roofline Plot

`plot_roofline.py` draws FP32 and FP64 kernel rooflines from Nsight Compute raw
CSV exports. It derives achieved FLOP/s and measured compute and DRAM ceilings
from the same report.

## Collect Input

Run a dedicated profiled experiment with `GPU_BENCH_PROFILE=ncu`, then export the
raw page from each `.ncu-rep` file:

```bash
ncu --import report.ncu-rep --csv --page raw > kernel_raw.csv
```

The report must contain the metrics from Nsight Compute's `roofline` set.
Profiled runs are not valid timing measurements.

## Plot

The script declares its matplotlib dependency through PEP 723 and runs directly
with `uv`:

```bash
uv run tools/roofline/plot_roofline.py kernel_raw.csv -o roofline.svg
```

Multiple CSV files may be passed. Launches with the same kernel name are
averaged; FP32 and FP64 points are shown only when the corresponding instruction
rate is meaningful.
