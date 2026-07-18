from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from model import (
    BASELINE_BACKEND,
    METRIC_SPECS,
    BackendSummary,
    GroupedMeasurements,
    GroupKey,
    Measurement,
    MetricName,
    MetricSpec,
    SummaryGroups,
    SummaryRow,
    Status,
)


def parse_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def preferred_metric(records: list[Measurement]) -> MetricSpec:
    records = [record for record in records if record.status == Status.OK and record.valid]
    for metric in (MetricName.USEC, MetricName.TIME_PER_ITER_S, MetricName.TIME_S):
        if any(metric.value in record.fields for record in records):
            return METRIC_SPECS[metric]
    return METRIC_SPECS[MetricName.USEC]


def metric_value_for(summary: BackendSummary, metric: MetricSpec) -> float | None:
    if metric.name == MetricName.GBYTES_PER_S:
        return summary.bandwidth
    return summary.metric_value


def metric_min_for(summary: BackendSummary, metric: MetricSpec) -> float | None:
    if metric.name == MetricName.GBYTES_PER_S:
        return None
    return summary.metric_min


def relative_delta(base: float | None, value: float | None) -> float | None:
    if base is None or value is None or base == 0:
        return None
    return (value - base) / base * 100.0


def relative_speedup(base: float | None, value: float | None, lower_is_better: bool) -> float | None:
    if base is None or value is None or base == 0 or value == 0:
        return None
    if lower_is_better:
        return base / value
    return value / base


@dataclass(frozen=True)
class SummaryTable:
    baseline: str
    metric_by_benchmark: dict[str, MetricSpec]
    groups: SummaryGroups

    @classmethod
    def from_measurements(
        cls,
        measurements: list[Measurement],
        baseline: str = BASELINE_BACKEND,
        metric_override: MetricName | None = None,
    ) -> "SummaryTable":
        grouped = cls._group_measurements(measurements)
        metric_by_benchmark = cls._select_metrics(grouped, metric_override)
        groups = cls._aggregate_groups(grouped, metric_by_benchmark)
        return cls(baseline=baseline, metric_by_benchmark=metric_by_benchmark, groups=groups)

    @staticmethod
    def _group_measurements(measurements: list[Measurement]) -> GroupedMeasurements:
        grouped: GroupedMeasurements = defaultdict(list)
        for measurement in measurements:
            key = (
                measurement.benchmark,
                measurement.topology,
                measurement.n,
                measurement.case,
                measurement.backend,
            )
            grouped[key].append(measurement)
        return dict(grouped)

    @staticmethod
    def _select_metrics(grouped: GroupedMeasurements, metric_override: MetricName | None) -> dict[str, MetricSpec]:
        if metric_override is not None:
            benchmarks = {benchmark for benchmark, _topology, _n, _case, _backend in grouped}
            return {benchmark: METRIC_SPECS[metric_override] for benchmark in benchmarks}

        records_by_benchmark: dict[str, list[Measurement]] = defaultdict(list)
        for (benchmark, _topology, _n, _case, _backend), records in grouped.items():
            records_by_benchmark[benchmark].extend(records)
        return {benchmark: preferred_metric(records) for benchmark, records in records_by_benchmark.items()}

    @staticmethod
    def _aggregate_groups(grouped: GroupedMeasurements, metric_by_benchmark: dict[str, MetricSpec]) -> SummaryGroups:
        groups: SummaryGroups = defaultdict(dict)
        for (benchmark, topology, n, case, backend), records in grouped.items():
            metric = metric_by_benchmark[benchmark]
            supported_records = [record for record in records if record.status == Status.OK and record.valid]
            metric_values = [
                value
                for value in (parse_float(record.fields.get(metric.name.value)) for record in supported_records)
                if value is not None
            ]
            bandwidths = [
                value
                for value in (
                    parse_float(record.fields.get(MetricName.GBYTES_PER_S.value)) for record in supported_records
                )
                if value is not None and value > 0.0
            ]
            nbytes = next(
                (
                    int(value)
                    for value in (parse_float(record.fields.get("bytes")) for record in supported_records)
                    if value is not None
                ),
                None,
            )
            if any(
                record.status == Status.ERROR
                or (record.status != Status.NOT_IMPLEMENTED and not record.valid)
                for record in records
            ):
                status = Status.ERROR
            elif supported_records:
                status = Status.OK
            else:
                status = Status.NOT_IMPLEMENTED
            groups[GroupKey(benchmark, topology, n, case)][backend] = BackendSummary(
                backend=backend,
                metric_value=mean_or_none(metric_values),
                metric_min=min(metric_values) if metric_values else None,
                bandwidth=mean_or_none(bandwidths),
                nbytes=nbytes,
                trials=len(supported_records) if supported_records else len(records),
                valid_all=status != Status.ERROR,
                status=status,
            )
        return {key: dict(value) for key, value in groups.items()}

    def benchmarks(self) -> list[str]:
        return sorted({key.benchmark for key in self.groups})

    def cases_for(self, benchmark: str) -> list[str]:
        return sorted({key.case for key in self.groups if key.benchmark == benchmark})

    def topologies_for(self, benchmark: str, case: str = "") -> list[str]:
        return sorted({key.topology for key in self.groups if key.benchmark == benchmark and key.case == case})

    def sizes_for(self, benchmark: str, topology: str, case: str = "") -> list[int]:
        return sorted(
            key.n
            for key in self.groups
            if key.benchmark == benchmark and key.topology == topology and key.case == case
        )

    def rows(self) -> list[SummaryRow]:
        rows: list[SummaryRow] = []
        for benchmark in self.benchmarks():
            for case in self.cases_for(benchmark):
                for topology in self.topologies_for(benchmark, case):
                    for n in self.sizes_for(benchmark, topology, case):
                        rows.extend(self.rows_for(benchmark, topology, n, case))
        return rows

    def rows_for(self, benchmark: str, topology: str, n: int, case: str = "") -> list[SummaryRow]:
        key = GroupKey(benchmark, topology, n, case)
        summaries = self.groups[key]
        metric = self.metric_by_benchmark[benchmark]
        base_summary = summaries.get(self.baseline)
        base_value = metric_value_for(base_summary, metric) if base_summary is not None else None
        rows = [self._row_from_summary(key, metric, summary, base_value) for summary in summaries.values()]
        return sorted(rows, key=self._row_sort_key)

    def _row_from_summary(
        self,
        key: GroupKey,
        metric: MetricSpec,
        summary: BackendSummary,
        base_value: float | None,
    ) -> SummaryRow:
        value = metric_value_for(summary, metric)
        return SummaryRow(
            benchmark=key.benchmark,
            topology=key.topology,
            n=key.n,
            metric=metric,
            backend=summary.backend,
            value=value,
            value_min=metric_min_for(summary, metric),
            bandwidth=summary.bandwidth,
            nbytes=summary.nbytes,
            delta_pct_vs_base=relative_delta(base_value, value),
            speedup_vs_base=relative_speedup(base_value, value, metric.lower_is_better),
            trials=summary.trials,
            valid_all=summary.valid_all,
            case=key.case,
            status=summary.status,
        )

    def _row_sort_key(self, row: SummaryRow) -> tuple[bool, float, str]:
        speedup = row.speedup_vs_base
        return (speedup is None, -(speedup or 0.0), row.backend)
