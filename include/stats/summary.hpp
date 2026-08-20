#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gpu_bench {

struct bench_stats {
  double min_s = 0.0;
  double max_s = 0.0;
  double avg_s = 0.0;
  double median_s = 0.0;
  double p25_s = 0.0;
  double p75_s = 0.0;
  // Sample standard deviation (n-1): these iterations are a sample of the
  // process, not the whole population.
  double stddev_s = 0.0;
  double total_s = 0.0;
  int iterations = 0;
  /* The raw per-iteration samples, in iteration order. Kept so that a caller
   * can reduce them across ranks iteration by iteration before summarizing;
   * summarizing first and reducing the summaries afterwards answers a different
   * question (see stats/collective.hpp). */
  std::vector<double> samples;
};

// Turns a series of per-iteration times into the distribution reported for it.
// Takes the vector by value: it needs a sorted copy for the percentiles, and
// the original order is preserved in the returned `samples`.
inline bench_stats summarize(std::vector<double> samples) {
  if (samples.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("timing sample count exceeds supported iteration count");
  }

  bench_stats stats;
  stats.iterations = static_cast<int>(samples.size());
  if (samples.empty()) {
    return stats;
  }

  double total = 0.0;
  for (const double sample : samples) {
    total += sample;
  }
  stats.total_s = total;
  stats.avg_s = total / static_cast<double>(samples.size());
  stats.min_s = *std::min_element(samples.begin(), samples.end());
  stats.max_s = *std::max_element(samples.begin(), samples.end());

  // Two-pass rather than the sum-of-squares shortcut: the variance is tiny
  // beside the mean here (samples cluster within a few percent of the mean),
  // so the shortcut loses most of its significant digits.
  if (samples.size() > 1U) {
    double sq = 0.0;
    for (const double sample : samples) {
      const double d = sample - stats.avg_s;
      sq += d * d;
    }
    stats.stddev_s = std::sqrt(sq / static_cast<double>(samples.size() - 1U));
  }

  stats.samples = samples;
  std::sort(samples.begin(), samples.end());
  const auto mid = samples.size() / 2;
  stats.median_s = samples.size() % 2 == 0 ? 0.5 * (samples[mid - 1U] + samples[mid]) : samples[mid];
  stats.p25_s = samples[samples.size() / 4U];
  stats.p75_s = samples[(3U * samples.size()) / 4U];
  return stats;
}

}  // namespace gpu_bench
