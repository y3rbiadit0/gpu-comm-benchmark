"""Loading and reshaping Benchscribe JSON.

The two `SUPPORTED_*_SCHEMA` constants are this tool's half of the contract with
Benchscribe. If Benchscribe bumps a `schema_version`, bump the matching constant
here and adjust the readers - do not widen the check.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

SUPPORTED_POINTS_SCHEMA = 1
SUPPORTED_FIT_SCHEMA = 1

TOPOLOGY_ORDER = ("1n1g", "1n2g", "1n4g", "2n1g", "2n4g")


class SchemaMismatch(Exception):
    """A JSON file this tool cannot read, with the reason spelled out."""


def load_json(path: Path, expected_schema: int, kind: str) -> dict:
    with path.open() as handle:
        payload = json.load(handle)
    version = payload.get("schema_version")
    if version != expected_schema:
        raise SchemaMismatch(
            f"{path} is {kind} schema_version {version!r}, this tool speaks "
            f"{expected_schema}. Regenerate it with the matching benchscribe."
        )
    return payload


def topology_ranks(topology: str) -> int | None:
    """Ranks implied by a topology label: `2n4g` -> 8. None if unparseable."""
    m = re.fullmatch(r"(\d+)n(\d+)g", topology)
    return int(m.group(1)) * int(m.group(2)) if m else None


def is_single_rank(topology: str) -> bool:
    """A single-rank topology communicates with nobody.

    `1n1g` runs the benchmark on one rank: alltoall becomes a local memcpy with
    zero bus bandwidth, moe's routing distributions are indistinguishable, and
    allreduce reduces one contribution. Those points are legitimate controls --
    cg_step's 1n1g isolates its compute term -- but they are not communication
    measurements, so plotting them beside real ones invites misreading.
    """
    return topology_ranks(topology) == 1


def topology_key(topology: str) -> tuple[int, str]:
    """Sort intra-node before inter-node, unknown topologies last."""
    try:
        return (TOPOLOGY_ORDER.index(topology), "")
    except ValueError:
        return (len(TOPOLOGY_ORDER), topology)


def format_bytes(value: float | int | None) -> str:
    if value is None:
        return "-"
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            if value >= 10 or value == int(value):
                return f"{value:.0f} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.0f} GiB"


def write_table(path: Path, header: list[str], rows: list[list]) -> None:
    """The table view that every figure ships beside it.

    Required, not a convenience: three of the light-mode series colours sit
    below 3:1 contrast on the chart surface, and the palette's relief rule says
    a chart using them must offer a non-colour way to read every value.
    """
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class Sweep:
    """Benchscribe points reshaped into (case, topology) -> backend -> curve."""

    def __init__(self, points: list[dict], benchmark: str | None = None,
                 include_single_rank: bool = False):
        from .theme import BACKEND_ORDER

        self._backend_order = BACKEND_ORDER
        self.excluded_topologies: list[str] = []
        self.metric = ""
        self.unit = ""
        self.curves: dict[tuple[str, str], dict[str, list[dict]]] = {}
        for point in points:
            if benchmark and point["benchmark"] != benchmark:
                continue
            if not include_single_rank and is_single_rank(point["topology"]):
                if point["topology"] not in self.excluded_topologies:
                    self.excluded_topologies.append(point["topology"])
                continue
            # Only OK/validated points carry meaning; benchscribe already flags
            # the rest, and a figure must not average a failed run into a curve.
            if not point.get("valid") or point.get("status") != "OK":
                continue
            if point.get("bytes") is None or point.get("value_mean") is None:
                continue
            self.metric = self.metric or point["metric"]
            self.unit = self.unit or point["unit"]
            key = (point["case"], point["topology"])
            self.curves.setdefault(key, {}).setdefault(point["backend"], []).append(point)
        for backends in self.curves.values():
            for series in backends.values():
                series.sort(key=lambda item: item["bytes"])

    @property
    def cases(self) -> list[str]:
        return sorted({case for case, _ in self.curves})

    @property
    def topologies(self) -> list[str]:
        return sorted({topo for _, topo in self.curves}, key=topology_key)

    def backends(self) -> list[str]:
        seen = {backend for backends in self.curves.values() for backend in backends}
        known = [backend for backend in self._backend_order if backend in seen]
        return known + sorted(seen.difference(self._backend_order))

    @staticmethod
    def band(point: dict) -> tuple[float, float] | None:
        """Job-to-job spread, which is the dispersion worth drawing.

        Trials inside one job share an allocation, so their agreement says
        little; `across_runs` is benchscribe's spread across independent jobs.
        """
        across = point.get("across_runs")
        if not across or across.get("p25") is None or across.get("p75") is None:
            return None
        if across.get("n_runs", 0) < 2:
            return None
        return (across["p25"], across["p75"])
