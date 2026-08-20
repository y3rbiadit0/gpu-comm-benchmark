#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "timing.hpp"

namespace gpu_bench {

/* How a communication step is timed across the ranks that take part in it.
 *
 * A step costs what its slowest participant took, so the cost of iteration i is
 * the max over ranks of that rank's iteration i - and the headline number is
 * the mean of those per-iteration maxima:
 *
 *   AVG_i( MAX_r t[r][i] )      not      MAX_r( AVG_i t[r][i] )
 *
 * The two agree only if the same rank is slowest every iteration. In practice
 * the straggler moves around, so MAX(AVG) averages away part of the very
 * imbalance the step actually pays for and reads faster than the run was.
 *
 * Reducing the whole series rather than the summaries also keeps one
 * distribution behind every reported field: min, max, median, p25, p75 and
 * stddev then all describe the same series of global iteration times, instead
 * of mixing a cross-rank headline with one rank's local spread.
 *
 * This header holds the definition; the transport that carries out the
 * element-wise max lives in the backend adapters (collective_stats_mpi.hpp,
 * collective_stats_shmem.hpp), which all expose it as `collective_stats(...)`.
 *
 * `max_across_ranks(in, out, count)` must fill `out` with the element-wise max
 * of every rank's `in`. */
template <typename MaxAcrossRanks>
inline bench_stats collective_stats(const bench_stats& local, MaxAcrossRanks&& max_across_ranks) {
  std::vector<double> global(local.samples.size());
  if (!global.empty()) {
    max_across_ranks(local.samples.data(), global.data(), global.size());
  }
  return summarize(std::move(global));
}

}  // namespace gpu_bench
