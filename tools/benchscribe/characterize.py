from __future__ import annotations

import statistics
from dataclasses import dataclass

from summary import SummaryTable


ALPHA_MAX_BYTES = 4 * 1024


@dataclass(frozen=True)
class Characterization:
    benchmark: str
    case: str
    topology: str
    backend: str
    unit: str                  # unit of alpha as reported (us or s)
    alpha: float | None        # median latency for messages up to ALPHA_MAX_BYTES
    binf_gbs: float | None     # asymptotic bus bandwidth: peak gbytes_per_s
    peak_bytes: int | None     # message size at peak bandwidth
    nhalf_bytes: float | None  # alpha * binf -> message size at half of peak
    tail_gbs: float | None     # bandwidth at the largest message (plateau / cliff)
    points: int                # number of distinct message sizes in the sweep


def characterize(table: SummaryTable) -> list[Characterization]:
    # Gather each backend's swept points, reusing the trial-averaged summaries
    # SummaryTable already computed. One point = (bytes, mean latency, mean GB/s).
    sweeps: dict[tuple[str, str, str, str], list[tuple[int, float, float]]] = {}
    units: dict[tuple[str, str], str] = {}
    for key, summaries in table.groups.items():
        units[(key.benchmark, key.case)] = table.metric_by_benchmark[key.benchmark].unit
        for backend, summary in summaries.items():
            if summary.nbytes is None or summary.metric_value is None or summary.bandwidth is None:
                continue
            sweeps.setdefault((key.benchmark, key.case, key.topology, backend), []).append(
                (summary.nbytes, summary.metric_value, summary.bandwidth)
            )

    results: list[Characterization] = []
    for (benchmark, case, topology, backend), points in sorted(sweeps.items()):
        points.sort()  # by message size ascending
        unit = units[(benchmark, case)]

        small_latencies = [
            latency for nbytes, latency, _bw in points if nbytes <= ALPHA_MAX_BYTES
        ]
        # Truncated sweeps may start above 4 KiB; retain a useful floor estimate
        # from their smallest available message rather than dropping the backend.
        alpha = statistics.median(small_latencies or [points[0][1]])
        peak = max(range(len(points)), key=lambda i: points[i][2])
        binf = points[peak][2]
        peak_bytes = points[peak][0]
        tail_gbs = points[-1][2]                                  # largest-message BW

        # n_half = alpha * binf, alpha in seconds and binf in bytes/s.
        # Reported alpha is microseconds ("us"), binf is GB/s, so:
        #   us * 1e-6  *  GB/s * 1e9  =  bytes * 1e3
        scale = 1e3 if unit == "us" else 1e9                      # "s" -> 1e9
        nhalf = alpha * binf * scale

        results.append(
            Characterization(benchmark, case, topology, backend, unit,
                             alpha, binf, peak_bytes, nhalf, tail_gbs, len(points))
        )
    return results
