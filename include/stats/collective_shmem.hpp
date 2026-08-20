#pragma once

#include <shmem.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "stats/collective.hpp"

namespace gpu_bench {

// Elements the gather buffer passed to collective_stats() must hold. Allocate
// it with shmem_malloc() so it is symmetric; `iterations` is the timed
// iteration count, identical on every PE.
inline std::size_t collective_gather_elements(int pes, int iterations) {
  if (pes <= 0 || iterations <= 0) {
    throw std::invalid_argument("collective statistics require positive PE and iteration counts");
  }
  const auto pe_count = static_cast<std::size_t>(pes);
  const auto iteration_count = static_cast<std::size_t>(iterations);
  if (pe_count > std::numeric_limits<std::size_t>::max() / sizeof(double) / iteration_count) {
    throw std::overflow_error("collective statistics gather size overflow");
  }
  return pe_count * iteration_count;
}

/* OpenSHMEM adapter for collective_stats(). OpenSHMEM's reductions want
 * symmetric source, destination and work buffers per call, and only PE 0
 * reports, so the series is gathered to PE 0 the same way these benchmarks
 * already gather their validation flags. `gather` must be symmetric memory of
 * collective_gather_elements(pes, iterations) doubles.
 *
 * Collective: every PE must call it with the same number of samples. PE 0 gets
 * the reduced stats; every other PE gets its own local stats back, which it has
 * no use for. */
inline bench_stats collective_stats(const bench_stats& local, double* gather, int pe, int pes) {
  if (pe < 0 || pes <= 0 || pe >= pes) {
    throw std::invalid_argument("invalid PE coordinates for collective statistics");
  }
  if (local.samples.empty()) {
    return local;
  }

  return collective_stats(local, [&](const double* in, double* out, std::size_t count) {
    shmem_putmem(gather + static_cast<std::size_t>(pe) * count, in, count * sizeof(double), 0);
    shmem_quiet();
    shmem_barrier_all();

    if (pe == 0) {
      std::copy(gather, gather + count, out);
      for (int source_pe = 1; source_pe < pes; ++source_pe) {
        const double* series = gather + static_cast<std::size_t>(source_pe) * count;
        for (std::size_t i = 0; i < count; ++i) {
          out[i] = std::max(out[i], series[i]);
        }
      }
    } else {
      std::copy(in, in + count, out);
    }
    // Nobody may overwrite the gather buffer before PE 0 has read it, and these
    // benchmarks reuse one buffer across a size sweep.
    shmem_barrier_all();
  });
}

}  // namespace gpu_bench
