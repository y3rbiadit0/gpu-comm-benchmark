from __future__ import annotations

import csv
import datetime as dt
from typing import TextIO

from model import SummaryRow
from summary import SummaryTable


def format_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def format_delta(row: SummaryRow, baseline: str) -> str:
    if row.backend == baseline and row.delta_pct_vs_base is not None:
        return "0.0%"
    if row.delta_pct_vs_base is None:
        return "-"
    return f"{row.delta_pct_vs_base:+.1f}%"


def format_speedup(row: SummaryRow, baseline: str) -> str:
    if row.backend == baseline and row.speedup_vs_base is not None:
        return "1.00x"
    if row.speedup_vs_base is None:
        return "-"
    return f"{row.speedup_vs_base:.2f}x"


def render_markdown(table: SummaryTable, out: TextIO) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out.write("# Benchmark Results Summary\n\n")
    out.write(f"_Generated {stamp}. Baseline: `{table.baseline}`._\n\n")
    out.write(
        "Latency metrics are lower-is-better; **Speedup** = baseline / backend "
        "for latency metrics and backend / baseline for bandwidth metrics. "
        "**Delta** is relative to baseline; negative latency deltas are faster.\n\n"
    )

    for benchmark in table.benchmarks():
        metric = table.metric_by_benchmark[benchmark]
        out.write(f"## {benchmark}\n\n")
        out.write(f"Primary metric: `{metric.name.value}` ({metric.unit}).\n\n")
        for topology in table.topologies_for(benchmark):
            out.write(f"### {topology}\n\n")
            out.write(
                f"| Size (n) | Bytes | Backend | {metric.name.value} ({metric.unit}) | "
                f"Min ({metric.unit}) | GB/s | Delta vs base | Speedup | Trials | Valid |\n"
            )
            out.write("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |\n")
            for n in table.sizes_for(benchmark, topology):
                for row in table.rows_for(benchmark, topology, n):
                    bytes_field = "-" if row.nbytes is None else str(row.nbytes)
                    out.write(
                        f"| {row.n} | {bytes_field} | `{row.backend}` | {format_number(row.value)} "
                        f"| {format_number(row.value_min)} | {format_number(row.bandwidth)} "
                        f"| {format_delta(row, table.baseline)} | {format_speedup(row, table.baseline)} "
                        f"| {row.trials} | {'PASS' if row.valid_all else 'FAIL'} |\n"
                    )
            out.write("\n")


def render_csv(table: SummaryTable, out: TextIO) -> None:
    writer = csv.writer(out)
    writer.writerow(
        [
            "benchmark",
            "topology",
            "n",
            "backend",
            "metric",
            "unit",
            "value_mean",
            "value_min",
            "gbytes_per_s",
            "delta_pct_vs_base",
            "speedup_vs_base",
            "trials",
            "valid",
        ]
    )
    for row in table.rows():
        writer.writerow(
            [
                row.benchmark,
                row.topology,
                row.n,
                row.backend,
                row.metric.name.value,
                row.metric.unit,
                "" if row.value is None else f"{row.value:.9g}",
                "" if row.value_min is None else f"{row.value_min:.9g}",
                "" if row.bandwidth is None else f"{row.bandwidth:.9g}",
                "" if row.delta_pct_vs_base is None else f"{row.delta_pct_vs_base:.3f}",
                "" if row.speedup_vs_base is None else f"{row.speedup_vs_base:.4f}",
                row.trials,
                "PASS" if row.valid_all else "FAIL",
            ]
        )
