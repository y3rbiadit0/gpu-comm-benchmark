"""Figure-pipeline tests.

These assert the contract and the guarantees that are easy to break silently -
schema checking, the fixed backend-to-colour map, the companion table view -
rather than pixel output.
"""

from __future__ import annotations

import csv
import json

import pytest

from gpu_bench_plot.cli import main
from gpu_bench_plot.data import SchemaMismatch, Sweep, format_bytes, load_json
from gpu_bench_plot.figures import select_size
from gpu_bench_plot.theme import BACKEND_ORDER, THEMES, colour_for


def make_point(backend, topology, case, n, usec, gbs, **overrides):
    point = {
        "benchmark": "halo_1d", "topology": topology, "case": case, "backend": backend,
        "n": n, "bytes": 16 * n, "metric": "usec", "unit": "us",
        "value_mean": usec, "value_min": usec * 0.95, "gbytes_per_s": gbs,
        "valid": True, "status": "OK",
        # One RunStats per independent job, each a full five-number summary -
        # this is what the dist figure turns into a box.
        "runs": [
            {"job": str(9000 + job), "mean": usec * shift, "median": usec * shift,
             "p25": usec * shift * 0.96, "p75": usec * shift * 1.05,
             "minimum": usec * shift * 0.92, "maximum": usec * shift * 1.11,
             "stddev": usec * 0.03, "iterations": 100}
            for job, shift in enumerate((0.98, 1.0, 1.03))
        ],
        "across_runs": {"n_runs": 3, "median": usec, "p25": usec * 0.97,
                        "p75": usec * 1.05, "stddev": usec * 0.03},
    }
    point.update(overrides)
    return point


def write_points(path, points):
    path.write_text(json.dumps({"schema_version": 1, "generated": "now", "points": points}))
    return path


@pytest.fixture
def points():
    out = []
    for topology in ("1n2g", "2n4g"):
        for case in ("isolated", "steady"):
            for backend, floor in (("cuda_mpi", 4.0), ("cuda_nccl", 9.0), ("cuda_nvshmem", 2.0)):
                for n in (1, 256, 65536):
                    usec = floor + n * 0.001
                    out.append(make_point(backend, topology, case, n, usec, 16 * n / usec / 1e3))
    return out


def test_schema_mismatch_names_both_versions(tmp_path):
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"schema_version": 99, "points": []}))
    with pytest.raises(SchemaMismatch) as excinfo:
        load_json(path, 1, "points")
    assert "99" in str(excinfo.value) and "speaks 1" in str(excinfo.value)


def test_invalid_and_errored_points_never_reach_a_curve(points):
    points = points + [
        make_point("oshmpi", "1n2g", "steady", 1, 5.0, 3.0, valid=False),
        make_point("oshmpi", "1n2g", "steady", 256, 5.0, 3.0, status="ERROR"),
    ]
    assert "oshmpi" not in Sweep(points).backends()


def test_backends_come_back_in_the_fixed_palette_order(points):
    sweep = Sweep(points)
    assert sweep.backends() == ["cuda_mpi", "cuda_nccl", "cuda_nvshmem"]


def test_colour_follows_the_backend_not_its_position():
    """A filtered-down figure must not repaint the survivors."""
    for theme in THEMES.values():
        everyone = {b: colour_for(b, theme) for b in BACKEND_ORDER}
        assert len(set(everyone.values())) == len(BACKEND_ORDER)
        assert colour_for("cuda_nvshmem", theme) == everyone["cuda_nvshmem"]
        # An unknown backend degrades to muted grey rather than stealing a slot.
        assert colour_for("some_future_backend", theme) == theme["muted"]


def test_band_needs_two_runs_to_mean_anything():
    assert Sweep.band({"across_runs": {"n_runs": 1, "p25": 1.0, "p75": 2.0}}) is None
    assert Sweep.band({"across_runs": {"n_runs": 3, "p25": 1.0, "p75": 2.0}}) == (1.0, 2.0)
    assert Sweep.band({}) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    # Whole sizes print without a decimal; only fractional ones carry .1f.
    [(None, "-"), (16, "16 B"), (1024, "1 KiB"), (1048576, "1 MiB"), (1536, "1.5 KiB")],
)
def test_format_bytes(value, expected):
    assert format_bytes(value) == expected


@pytest.mark.parametrize("theme", list(THEMES))
def test_every_figure_ships_a_table_view(tmp_path, points, theme):
    source = write_points(tmp_path / "points.json", points)
    outdir = tmp_path / "figures"
    assert main(["--points", str(source), "--outdir", str(outdir), "--theme", theme]) == 0

    for stem in ("halo_1d-sweep", "halo_1d-bandwidth", "halo_1d-speedup", "halo_1d-dist"):
        assert (outdir / f"{stem}.svg").stat().st_size > 0
        table = outdir / f"{stem}.csv"
        rows = list(csv.DictReader(table.open()))
        assert rows, f"{table} is empty"


def test_speedup_is_relative_to_the_baseline(tmp_path, points):
    source = write_points(tmp_path / "points.json", points)
    outdir = tmp_path / "figures"
    assert main(["--points", str(source), "--figure", "heatmap", "--outdir", str(outdir)]) == 0

    rows = list(csv.DictReader((outdir / "halo_1d-speedup.csv").open()))
    baseline = [r for r in rows if r["backend"] == "cuda_mpi"]
    assert baseline and all(float(r["speedup_vs_baseline"]) == 1.0 for r in baseline)
    # cuda_nvshmem has the lower latency floor in the fixture, so it must be >1.
    faster = [r for r in rows if r["backend"] == "cuda_nvshmem" and int(r["bytes"]) == 16]
    assert faster and all(float(r["speedup_vs_baseline"]) > 1.0 for r in faster)


def test_fit_figure_is_required_only_when_asked_for_by_name(tmp_path, points, capsys):
    source = write_points(tmp_path / "points.json", points)
    outdir = tmp_path / "figures"

    assert main(["--points", str(source), "--figure", "fit", "--outdir", str(outdir)]) == 2
    # Without --fit, "all" still produces everything else rather than bailing out.
    assert main(["--points", str(source), "--figure", "all", "--outdir", str(outdir)]) == 0
    assert (outdir / "halo_1d-sweep.svg").exists()
    assert not (outdir / "halo_1d-fit.svg").exists()
    assert "skipping fit figure" in capsys.readouterr().err


def test_fit_figure_renders_from_fit_json(tmp_path, points):
    source = write_points(tmp_path / "points.json", points)
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({
        "schema_version": 1, "generated": "now", "alpha_max_bytes": 4096,
        "fits": [
            {"benchmark": "halo_1d", "case": "steady", "topology": "1n2g",
             "backend": backend, "unit": "us", "alpha": alpha, "binf_gbs": 200.0,
             "peak_bytes": 1048576, "nhalf_bytes": alpha * 200.0 * 1e3,
             "tail_gbs": 190.0, "points": 3}
            for backend, alpha in (("cuda_mpi", 4.0), ("cuda_nvshmem", 2.0))
        ],
    }))
    outdir = tmp_path / "figures"
    assert main(["--points", str(source), "--fit", str(fit),
                 "--figure", "fit", "--outdir", str(outdir)]) == 0
    rows = list(csv.DictReader((outdir / "halo_1d-fit.csv").open()))
    assert {r["backend"] for r in rows} == {"cuda_mpi", "cuda_nvshmem"}


def test_no_matching_benchmark_is_an_error(tmp_path, points):
    source = write_points(tmp_path / "points.json", points)
    assert main(["--points", str(source), "--benchmark", "nope",
                 "--outdir", str(tmp_path / "f")]) == 1


def test_select_size_snaps_to_a_swept_size(points):
    sweep = Sweep(points)
    assert select_size(sweep, "min") == 16          # n=1   -> 16 bytes
    assert select_size(sweep, "max") == 1048576     # n=65536
    # An arbitrary byte count lands on the nearest size actually measured,
    # rather than silently producing an empty figure.
    assert select_size(sweep, "5000") == 4096       # n=256
    assert select_size(Sweep([]), "min") is None


def test_dist_reports_a_five_number_summary_per_job(tmp_path, points):
    source = write_points(tmp_path / "points.json", points)
    outdir = tmp_path / "figures"
    assert main(["--points", str(source), "--figure", "dist", "--outdir", str(outdir)]) == 0

    rows = list(csv.DictReader((outdir / "halo_1d-dist.csv").open()))
    assert rows
    assert all(int(row["bytes"]) == 16 for row in rows), "dist should default to the smallest size"
    for row in rows:
        low, q1, med, q3, high = (
            float(row[k]) for k in ("min", "p25", "median", "p75", "max")
        )
        assert low <= q1 <= med <= q3 <= high


def test_dist_skips_runs_without_quartiles(tmp_path):
    """Results predating the percentile fields must not be guessed at."""
    bare = [
        make_point("cuda_mpi", "1n2g", "steady", n, 4.0 + n, 1.0,
                   runs=[{"job": "1", "mean": 4.0 + n}], across_runs=None)
        for n in (1, 256)
    ]
    source = write_points(tmp_path / "points.json", bare)
    outdir = tmp_path / "figures"
    assert main(["--points", str(source), "--figure", "dist", "--outdir", str(outdir)]) == 0
    assert not (outdir / "halo_1d-dist.svg").exists()


def test_dist_uses_the_requested_size(tmp_path, points, capsys):
    source = write_points(tmp_path / "points.json", points)
    outdir = tmp_path / "figures"
    assert main(["--points", str(source), "--figure", "dist", "--size", "max",
                 "--outdir", str(outdir)]) == 0
    assert "1048576 bytes" in capsys.readouterr().err
    rows = list(csv.DictReader((outdir / "halo_1d-dist.csv").open()))
    assert {int(row["bytes"]) for row in rows} == {1048576}


def test_caseless_benchmarks_get_clean_labels():
    """allreduce, alltoall, cg_step and pingpong emit no `case=` field."""
    from gpu_bench_plot.figures import axis_label, panel_title

    assert panel_title("", "1n4g") == "1n4g"
    assert panel_title("steady", "1n4g") == "steady · 1n4g"
    assert axis_label("", "usec") == "usec"
    assert axis_label("steady", "usec") == "steady\nusec"


def test_a_caseless_benchmark_renders_end_to_end(tmp_path):
    points = [
        make_point(backend, "1n4g", "", n, floor + n * 0.001, 1.0, benchmark="allreduce")
        for backend, floor in (("cuda_mpi", 4.0), ("cuda_nccl", 9.0))
        for n in (1, 256, 65536)
    ]
    source = write_points(tmp_path / "points.json", points)
    outdir = tmp_path / "figures"
    assert main(["--points", str(source), "--figure", "all", "--outdir", str(outdir)]) == 0
    for stem in ("allreduce-sweep", "allreduce-bandwidth", "allreduce-dist", "allreduce-speedup"):
        assert (outdir / f"{stem}.svg").stat().st_size > 0
        assert (outdir / f"{stem}.csv").stat().st_size > 0


def test_multiple_benchmarks_get_separate_figure_sets(tmp_path, points):
    """A results tree holds several benchmarks; they must not share panels."""
    other = [
        make_point(backend, "1n4g", "", n, 3.0 + n * 0.002, 2.0, benchmark="allreduce")
        for backend in ("cuda_mpi", "cuda_nccl")
        for n in (1, 256, 65536)
    ]
    source = write_points(tmp_path / "points.json", points + other)
    outdir = tmp_path / "figures"
    assert main(["--points", str(source), "--figure", "sweep", "--outdir", str(outdir)]) == 0

    assert (outdir / "halo_1d-sweep.svg").exists()
    assert (outdir / "allreduce-sweep.svg").exists()

    halo = list(csv.DictReader((outdir / "halo_1d-sweep.csv").open()))
    allreduce = list(csv.DictReader((outdir / "allreduce-sweep.csv").open()))
    # The caseless allreduce rows must not have leaked into the halo_1d table.
    assert {row["case"] for row in halo} == {"isolated", "steady"}
    assert {row["case"] for row in allreduce} == {""}


def test_unknown_benchmark_lists_what_is_available(tmp_path, points, capsys):
    source = write_points(tmp_path / "points.json", points)
    assert main(["--points", str(source), "--benchmark", "nope",
                 "--outdir", str(tmp_path / "f")]) == 1
    assert "halo_1d" in capsys.readouterr().err
