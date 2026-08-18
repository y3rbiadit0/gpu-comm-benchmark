#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

namespace gpu_bench {

class wall_timer {
 public:
  wall_timer() : start_(clock::now()) {}

  double seconds() const {
    return std::chrono::duration<double>(clock::now() - start_).count();
  }

 private:
  using clock = std::chrono::steady_clock;
  clock::time_point start_;
};

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
};

// Runs `body` for `warmup_iterations` untimed calls, then `timed_iterations`
// timed calls, returning the local timing distribution. `body` must perform one
// fully-completed communication step (including any device synchronization), so
// each recorded sample reflects a finished operation.
template <typename Body>
bench_stats run_benchmark(int warmup_iterations, int timed_iterations, Body&& body) {
  for (int i = 0; i < warmup_iterations; ++i) {
    body();
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(timed_iterations > 0 ? timed_iterations : 0));
  for (int i = 0; i < timed_iterations; ++i) {
    wall_timer timer;
    body();
    samples.push_back(timer.seconds());
  }

  bench_stats stats;
  stats.iterations = timed_iterations;
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

  std::sort(samples.begin(), samples.end());
  const auto mid = samples.size() / 2;
  stats.median_s = samples.size() % 2 == 0 ? 0.5 * (samples[mid - 1U] + samples[mid]) : samples[mid];
  stats.p25_s = samples[samples.size() / 4U];
  stats.p75_s = samples[(3U * samples.size()) / 4U];
  return stats;
}

}  // namespace gpu_bench
