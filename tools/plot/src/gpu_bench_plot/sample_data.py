"""Synthesize a results/ tree of halo_1d report lines from an alpha-beta model.

SYNTHETIC DATA. Nothing here was measured on any machine; the numbers come from
a hand-written alpha-beta model with a little noise. It exists so the figure
pipeline can be exercised end to end without a cluster allocation - never cite,
publish, or commit its output as a result.

The emitted lines follow the schema in the repository root README, so
benchscribe parses them exactly as it parses real Slurm output.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# (alpha_us, Binf_GB/s) per backend, for intra-node and inter-node paths.
INTRA = {
    "cuda_mpi": (4.2, 210.0), "cuda_nccl": (9.5, 240.0),
    "cuda_nvshmem": (2.1, 250.0), "oshmpi": (5.8, 120.0),
    "sycl_mpi": (4.6, 200.0), "sycl_oneccl": (11.0, 230.0),
}
INTER = {
    "cuda_mpi": (7.8, 22.0), "cuda_nccl": (14.0, 24.0),
    "cuda_nvshmem": (3.6, 18.0), "oshmpi": (9.9, 14.0),
    "sycl_mpi": (8.4, 21.0), "sycl_oneccl": (16.0, 23.0),
}
TOPOLOGIES = {"1n2g": (INTRA, 2), "1n4g": (INTRA, 4), "2n1g": (INTER, 2), "2n4g": (INTER, 8)}
HALO_WIDTHS = [1 << k for k in range(0, 21, 2)]
JOBS = 3  # independent allocations, so `across_runs` has something to describe


def write_tree(out_root: Path, seed: int = 7) -> int:
    rng = random.Random(seed)
    files = 0
    for topology, (params, ranks) in TOPOLOGIES.items():
        for backend, (alpha, binf) in params.items():
            for job in range(JOBS):
                job_skew = rng.uniform(0.97, 1.09)  # allocation-to-allocation drift
                lines = []
                for halo in HALO_WIDTHS:
                    nbytes = 16 * halo
                    for case, iters in (("isolated", 1), ("steady", 100)):
                        # Isolated pays the full submission cost; steady amortizes it.
                        floor = alpha if case == "isolated" else alpha * 0.45
                        usec = (floor + (nbytes / (binf * 1e9)) * 1e6) * job_skew
                        usec *= rng.uniform(0.98, 1.02)
                        low, high = usec * 0.94, usec * 1.18
                        gbs = nbytes / (usec * 1e-6) / 1e9
                        lines.append(
                            f"{backend}_halo_1d n={halo} ranks={ranks} bytes={nbytes} "
                            f"iters={iters} warmup=20 time_per_iter_s={usec * 1e-6:.9g} "
                            f"usec={usec:.6g} min_usec={low:.6g} max_usec={high:.6g} "
                            f"gbytes_per_s={gbs:.6g} median_usec={usec * 1.01:.6g} "
                            f"p25_usec={usec * 0.97:.6g} p75_usec={usec * 1.06:.6g} "
                            f"stddev_usec={usec * 0.04:.6g} case={case} timing=batch "
                            f"batch_iters={iters} "
                            f"batch_samples={100 if iters == 1 else 10} "
                            f"status=OK validation=PASS"
                        )
                directory = out_root / f"halo-1d-{backend.replace('_', '-')}-{topology}" / "halo_1d"
                directory.mkdir(parents=True, exist_ok=True)
                name = f"halo_1d_{backend}_{topology}-{9000 + job}-1-stdout.txt"
                (directory / name).write_text("\n".join(lines) + "\n")
                files += 1
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gpu-bench-plot-sample",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("out_root", type=Path, help="directory to write the synthetic tree into")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    files = write_tree(args.out_root, args.seed)
    print(f"wrote {files} synthetic stdout files to {args.out_root}")
    print("SYNTHETIC DATA - do not cite, publish, or commit these numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
