"""Matplotlib import with the non-interactive backend selected first.

`matplotlib.use` has to run before pyplot is imported, so every module in this
package takes pyplot from here rather than importing it directly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt"]
