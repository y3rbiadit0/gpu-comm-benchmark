"""Figures for gpu-comm-bench results.

Input is Benchscribe JSON and nothing else. This package does not parse Slurm
output and does not compute the alpha-beta fit: Benchscribe owns both, and the
`schema_version` field in each file is the contract between the two tools.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
