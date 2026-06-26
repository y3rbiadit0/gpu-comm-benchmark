from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path
from typing import TextIO

from model import KNOWN_BACKEND_NAMES, Measurement, ParsedReportLine


TOPOLOGY_RE = re.compile(r"(\d+n\d+g)")


def split_backend_and_benchmark(name: str) -> tuple[str, str] | None:
    for backend in KNOWN_BACKEND_NAMES:
        if name == backend:
            return backend, ""
        if name.startswith(f"{backend}_"):
            return backend, name[len(backend) + 1 :]
    return None


def parse_report_line(line: str) -> ParsedReportLine | None:
    line = line.strip()
    if "validation=" not in line or "=" not in line:
        return None
    try:
        tokens = shlex.split(line)
    except ValueError:
        return None
    if not tokens or "=" in tokens[0]:
        return None

    fields: dict[str, str] = {}
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if separator:
            fields[key] = value
    if "validation" not in fields:
        return None
    return ParsedReportLine(name=tokens[0], fields=fields)


def topology_from_path(path: Path) -> str:
    match = TOPOLOGY_RE.search(str(path))
    return match.group(1) if match else "unknown"


def parse_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def scan_results(results_dir: Path, warnings: TextIO = sys.stderr) -> list[Measurement]:
    measurements: list[Measurement] = []
    for path in sorted(results_dir.glob("**/*-stdout.txt")):
        topology = topology_from_path(path)
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as error:
            print(f"warning: cannot read {path}: {error}", file=warnings)
            continue

        for line in lines:
            parsed = parse_report_line(line)
            if parsed is None:
                continue
            split = split_backend_and_benchmark(parsed.name)
            if split is None:
                continue
            backend, benchmark = split
            measurements.append(
                Measurement(
                    backend=backend,
                    benchmark=benchmark,
                    topology=topology,
                    n=parse_int(parsed.fields.get("n")),
                    fields=parsed.fields,
                    valid=parsed.fields.get("validation", "FAIL").upper() == "PASS",
                )
            )
    return measurements
