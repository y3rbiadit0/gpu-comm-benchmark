#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Roofline chart from Nsight Compute raw-page CSV exports.

Input files come from profiled experiment runs (GPU_BENCH_PROFILE=ncu, --set roofline):

    ncu --import report.ncu-rep --csv --page raw > kernel_raw.csv

Each CSV may hold several kernel launches; launches of the same kernel are
averaged. Achieved FLOP/s follows ncu's own FP roofline recipe
(adds + muls + 2*FMAs, per-cycle rates times the measured clock), and the
compute/memory ceilings are the measured peak_sustained rates from the same
report, so points and roofs are consistent with the Nsight Compute GUI chart.

Usage:
    uv run tools/roofline/plot_roofline.py docs/analysis/data/*.csv \
        -o docs/analysis/roofline.svg
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

# Multipliers for the unit prefixes ncu emits in the units row (e.g. "Ghz",
# "Gbyte/s", "Kbyte/cycle"). Base units (inst/cycle, byte/s) scale by 1.
PREFIX = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}

FP64_RATES = [
    ("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed", 1),
    ("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed", 1),
    ("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed", 2),
]
FP32_RATES = [
    ("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed", 1),
    ("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed", 1),
    ("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed", 2),
]
SM_HZ = "sm__cycles_elapsed.avg.per_second"
DRAM_HZ = "dram__cycles_elapsed.avg.per_second"
DRAM_BW = "dram__bytes.sum.per_second"
PEAK_DRAM = "dram__bytes.sum.peak_sustained"      # bytes/cycle (dram clock)
PEAK_DFMA = "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained"
PEAK_FFMA = "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"
DURATION = "gpu__time_duration.sum"


def unit_scale(unit: str) -> float:
    unit = unit.strip()
    if unit == "us":
        return 1e-6
    if unit == "ns":
        return 1e-9
    if unit == "ms":
        return 1e-3
    return PREFIX.get(unit[:1], 1.0) if unit and unit[0] in PREFIX else 1.0


def load_rows(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        sys.exit(f"{path}: expected header + units + data rows")
    hdr, units = rows[0], rows[1]
    col = {h: i for i, h in enumerate(hdr)}
    for name in (SM_HZ, DRAM_HZ, DRAM_BW, PEAK_DRAM, PEAK_DFMA, PEAK_FFMA):
        if name not in col:
            sys.exit(f"{path}: missing metric {name} -- was this exported with "
                     f"--page raw from a --set roofline report?")

    def get(row, name):
        raw = row[col[name]].replace(",", "")
        return float(raw) * unit_scale(units[col[name]]) if raw else 0.0

    return [(row, get) for row in rows[2:] if row[col["Kernel Name"]]], col


def short_kernel_name(full: str) -> str:
    return full.split("(")[0].replace("<unnamed>::", "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csvs", nargs="+", type=Path, help="ncu raw-page CSV exports")
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("roofline.svg"), help="output figure (.svg/.png/.pdf)")
    ap.add_argument("--title", default="Kernel roofline (A100-SXM-64GB, measured ceilings)")
    args = ap.parse_args()

    # kernel points keyed by (file label, kernel, precision) -> list of (ai, flops)
    points = defaultdict(list)
    peaks = {}
    for path in args.csvs:
        rows, col = load_rows(path)
        label = path.stem.replace("_raw", "")
        for row, get in rows:
            kname = short_kernel_name(row[col["Kernel Name"]])
            sm_hz = get(row, SM_HZ)
            bw = get(row, DRAM_BW)
            if not peaks:
                peaks = {
                    "fp64": 2 * get(row, PEAK_DFMA) * sm_hz,
                    "fp32": 2 * get(row, PEAK_FFMA) * sm_hz,
                    "bw": get(row, PEAK_DRAM) * get(row, DRAM_HZ),
                }
            for prec, rates in (("fp64", FP64_RATES), ("fp32", FP32_RATES)):
                flops = sum(w * get(row, m) for m, w in rates) * sm_hz
                # skip the precision a kernel doesn't meaningfully use
                if flops > 0.005 * max(
                    sum(w * get(row, m) for m, w in r) * sm_hz
                    for r in (FP64_RATES, FP32_RATES)
                ) and flops > 0 and bw > 0:
                    points[(label, kname, prec)].append((flops / bw, flops))

    if not points:
        sys.exit("no kernel data points found")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ais = [ai for pts in points.values() for ai, _ in pts]
    x = [min(min(ais) / 8, 0.01), max(max(ais) * 8, peaks["fp32"] / peaks["bw"] * 4)]
    xs = [x[0] * (x[1] / x[0]) ** (i / 256) for i in range(257)]

    for prec, style in (("fp64", "--"), ("fp32", "-")):
        roof = peaks[prec]
        ax.plot(xs, [min(ai * peaks["bw"], roof) / 1e9 for ai in xs],
                style, color="0.35", lw=1.2)
        ax.annotate(f"{prec.upper()} peak {roof / 1e12:.1f} TFLOP/s",
                    xy=(x[1], roof / 1e9), xytext=(-6, 4),
                    textcoords="offset points", ha="right", fontsize=8, color="0.25")
    ridge = peaks["fp64"] / peaks["bw"]
    ax.annotate(f"HBM2e {peaks['bw'] / 1e12:.2f} TB/s",
                xy=(ridge / 6, ridge / 6 * peaks["bw"] / 1e9), fontsize=8,
                color="0.25", rotation=38, rotation_mode="anchor",
                xytext=(0, 6), textcoords="offset points")

    markers = {"fp32": "o", "fp64": "s"}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series = {}
    for (label, kname, prec), pts in sorted(points.items()):
        ai = sum(p[0] for p in pts) / len(pts)
        gf = sum(p[1] for p in pts) / len(pts) / 1e9
        key = f"{label}: {kname}"
        color = series.setdefault(key, colors[len(series) % len(colors)])
        ax.plot(ai, gf, markers[prec], color=color, ms=8, mec="black", mew=0.5,
                label=f"{key} ({prec}, {gf:.0f} GFLOP/s)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / DRAM byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title(args.title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    print(f"ceilings: FP64 {peaks['fp64']/1e12:.2f} TFLOP/s, "
          f"FP32 {peaks['fp32']/1e12:.2f} TFLOP/s, DRAM {peaks['bw']/1e12:.2f} TB/s")
    for (label, kname, prec), pts in sorted(points.items()):
        ai = sum(p[0] for p in pts) / len(pts)
        gf = sum(p[1] for p in pts) / len(pts)
        pct_bw = gf / ai / peaks["bw"] * 100
        print(f"  {label}: {kname} [{prec}] AI={ai:.3f} FLOP/B, "
              f"{gf/1e9:.1f} GFLOP/s, {pct_bw:.0f}% of DRAM roof "
              f"({len(pts)} launch{'es' if len(pts) > 1 else ''})")


if __name__ == "__main__":
    main()
