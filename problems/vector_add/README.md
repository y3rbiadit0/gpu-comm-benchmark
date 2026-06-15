# Vector Add

Purpose: basic distributed execution.

Each rank owns a contiguous chunk of the global vectors and computes `c[i] = a[i] + b[i]` on its local accelerator.
